"""Configuration file manager with atomic read/write behavior."""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from dotenv import dotenv_values

_ASSIGNMENT_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_FALLBACK_REWRITE_ERRNOS = {errno.EBUSY, errno.EXDEV}

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage `.env` read/write operations with optimistic versioning."""

    def __init__(self, env_path: Optional[Path] = None):
        self._env_path = env_path or self._resolve_env_path()
        self._lock = threading.RLock()

    @property
    def env_path(self) -> Path:
        """Return active `.env` path."""
        return self._env_path

    def read_config_map(self) -> Dict[str, str]:
        """Read key-value mapping from `.env` file."""
        if not self._env_path.exists():
            return {}

        values = dotenv_values(self._env_path)
        return {
            str(key): "" if value is None else str(value)
            for key, value in values.items()
            if key is not None
        }

    def get_config_version(self) -> str:
        """Return deterministic version string based on file state."""
        if not self._env_path.exists():
            return "missing:0"

        content = self._env_path.read_bytes()
        file_stat = self._env_path.stat()
        content_hash = hashlib.sha256(content).hexdigest()
        return f"{file_stat.st_mtime_ns}:{content_hash}"

    def get_updated_at(self) -> Optional[str]:
        """Return `.env` last update time in ISO8601 format."""
        if not self._env_path.exists():
            return None

        file_stat = self._env_path.stat()
        updated_at = datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc)
        return updated_at.isoformat()

    def get_aggregated_config_version(self) -> str:
        """返回所有受管理配置文件的聚合版本哈希。"""
        versions = [self.get_config_version()]
        # YAML 文件版本将由 SystemConfigService 注册后填充
        return hashlib.sha256("|".join(versions).encode()).hexdigest()

    def apply_updates(
        self,
        updates: Iterable[Tuple[str, str]],
        sensitive_keys: Set[str],
        mask_token: str,
    ) -> Tuple[List[str], List[str], str]:
        """Apply updates into `.env` file using atomic replace when possible."""
        with self._lock:
            current_values = self.read_config_map()
            mutable_updates: Dict[str, str] = {}
            skipped_masked: List[str] = []

            for key, value in updates:
                key_upper = key.upper()
                current_value = current_values.get(key_upper)

                if key_upper in sensitive_keys and value == mask_token:
                    if current_value not in (None, ""):
                        skipped_masked.append(key_upper)
                    continue

                if current_value == value:
                    continue

                mutable_updates[key_upper] = value

            if mutable_updates:
                self._atomic_upsert(mutable_updates)

            return list(mutable_updates.keys()), skipped_masked, self.get_config_version()

    def _atomic_upsert(self, updates: Dict[str, str]) -> None:
        """Write updates with atomic rename and in-place fallback for mounted files."""
        lines = self._read_lines()
        key_to_index = self._find_last_key_indexes(lines)

        for key, value in updates.items():
            line_value = value.replace("\n", "")
            new_line = f"{key}={line_value}"
            if key in key_to_index:
                lines[key_to_index[key]] = new_line
            else:
                lines.append(new_line)

        if not self._env_path.parent.exists():
            self._env_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = self._env_path.with_suffix(self._env_path.suffix + ".tmp")
        content = "\n".join(lines)
        if content and not content.endswith("\n"):
            content += "\n"

        with temp_path.open("w", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write(content)
            file_obj.flush()
            os.fsync(file_obj.fileno())

        try:
            os.replace(temp_path, self._env_path)
        except OSError as exc:
            if exc.errno not in _FALLBACK_REWRITE_ERRNOS:
                raise

            logger.warning(
                "Atomic replace for .env failed with errno=%s, falling back to in-place rewrite",
                exc.errno,
            )
            self._rewrite_in_place(content)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _rewrite_in_place(self, content: str) -> None:
        """Rewrite `.env` content in place when rename is unsupported by mount type."""
        with self._env_path.open("w", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write(content)
            file_obj.flush()
            os.fsync(file_obj.fileno())

    def _read_lines(self) -> List[str]:
        if not self._env_path.exists():
            return []
        return self._env_path.read_text(encoding="utf-8").splitlines()

    @staticmethod
    def _find_last_key_indexes(lines: List[str]) -> Dict[str, int]:
        key_to_index: Dict[str, int] = {}
        for index, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            matched = _ASSIGNMENT_PATTERN.match(raw_line)
            if not matched:
                continue

            key_to_index[matched.group(1).upper()] = index

        return key_to_index

    @staticmethod
    def _resolve_env_path() -> Path:
        env_file = os.getenv("ENV_FILE")
        if env_file:
            return Path(env_file).resolve()

        return (Path(__file__).resolve().parent.parent.parent / ".env").resolve()


class YamlConfigManager:
    """管理 YAML 配置文件，支持扁平化键值对读写。"""

    def __init__(self, yaml_path: Path):
        self._yaml_path = yaml_path

    def read_config_map(self) -> Dict[str, str]:
        """读取 YAML 文件并返回扁平化的键值对。"""
        if yaml is None:
            return {}
        if not self._yaml_path.exists():
            return {}
        try:
            with open(self._yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return self._flatten_dict(data)
        except Exception:
            return {}

    def write_config_map(self, flat_map: Dict[str, str]) -> bool:
        """将扁平化键值对写入嵌套 YAML 文件（与现有内容合并，非覆盖）。"""
        if yaml is None:
            return False
        temp_path = self._yaml_path.with_suffix(self._yaml_path.suffix + ".tmp")
        try:
            # 读取现有文件内容
            existing: dict = {}
            if self._yaml_path.exists():
                with open(self._yaml_path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
            # 反扁平化变更内容
            changes = self._unflatten_dict(flat_map)
            # 深度合并变更到现有内容
            merged = self._deep_merge(existing, changes)
            if not self._yaml_path.parent.exists():
                self._yaml_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "w", encoding="utf-8", newline="\n") as f:
                yaml.safe_dump(merged, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self._yaml_path)
            return True
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            return False

    def get_version(self) -> str:
        """返回此 YAML 文件的版本哈希。"""
        if not self._yaml_path.exists():
            return "missing:0"
        content = self._yaml_path.read_bytes()
        file_stat = self._yaml_path.stat()
        content_hash = hashlib.sha256(content).hexdigest()
        return f"{file_stat.st_mtime_ns}:{content_hash}"

    @staticmethod
    def _flatten_dict(data: dict, prefix: str = "") -> Dict[str, str]:
        """将嵌套字典扁平化为点分隔的键。"""
        result: Dict[str, str] = {}
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(YamlConfigManager._flatten_dict(value, full_key))
            elif isinstance(value, list):
                result[full_key] = ",".join(str(v) for v in value)
            elif value is None:
                result[full_key] = ""
            else:
                result[full_key] = str(value)
        return result

    @staticmethod
    def _unflatten_dict(flat_map: Dict[str, str]) -> dict:
        """将扁平点分隔键转换回嵌套字典。"""
        result: dict = {}
        for key, value in flat_map.items():
            parts = key.split(".")
            current = result
            for i, part in enumerate(parts[:-1]):
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = YamlConfigManager._coerce_value(value)
        return result

    @staticmethod
    def _coerce_value(value: str):
        """将字符串值转换为适当的 Python 类型。"""
        if not value.strip():
            return None
        lower = value.strip().lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @staticmethod
    def _deep_merge(base: dict, updates: dict) -> dict:
        """深度合并两个字典，updates 中的值覆盖 base 中的值。"""
        result = dict(base)
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = YamlConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
