# -*- coding: utf-8 -*-
"""System configuration service for `.env` based settings."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.config import Config, setup_env
from src.core.config_manager import ConfigManager, YamlConfigManager
from src.core.config_registry import (
    build_schema_response,
    get_category_definitions,
    get_field_definition,
    get_registered_field_keys,
)

logger = logging.getLogger(__name__)

# YAML config files managed by the settings page
_MANAGED_YAML_FILES = [
    "config/scorer_config.yaml",
    "config/industry_percentiles.yaml",
    "stock_selector/backtest_config.yaml",
    "watchdog/strategies/intraday_t0_config.yaml",
]


class ConfigValidationError(Exception):
    """Raised when one or more submitted fields fail validation."""

    def __init__(self, issues: List[Dict[str, Any]]):
        super().__init__("Configuration validation failed")
        self.issues = issues


class ConfigConflictError(Exception):
    """Raised when submitted config_version is stale."""

    def __init__(self, current_version: str):
        super().__init__("Configuration version conflict")
        self.current_version = current_version


class SystemConfigService:
    """Service layer for reading, validating, and updating runtime configuration."""

    def __init__(self, manager: Optional[ConfigManager] = None, yaml_managers: Optional[Dict[str, YamlConfigManager]] = None):
        self._manager = manager or ConfigManager()
        self._yaml_managers: Dict[str, YamlConfigManager] = yaml_managers or {}
        if not self._yaml_managers:
            project_root = Path(__file__).resolve().parent.parent.parent
            for yaml_rel_path in _MANAGED_YAML_FILES:
                yaml_path = project_root / yaml_rel_path
                if yaml_path.exists():
                    self._yaml_managers[yaml_rel_path] = YamlConfigManager(yaml_path)

    def get_schema(self) -> Dict[str, Any]:
        """Return grouped schema metadata for UI rendering."""
        return build_schema_response()

    def get_config(self, include_schema: bool = True, mask_token: str = "******") -> Dict[str, Any]:
        """Return current config values from .env and managed YAML files."""
        config_map = self._manager.read_config_map()
        registered_keys = set(get_registered_field_keys())

        # 读取 YAML 配置文件中的值
        for yaml_path, yaml_mgr in self._yaml_managers.items():
            yaml_map = yaml_mgr.read_config_map()
            for yaml_key, yaml_value in yaml_map.items():
                full_key = f"{yaml_path}:{yaml_key}"
                config_map[full_key] = yaml_value

        all_keys = set(config_map.keys()) | {
            k for k in registered_keys if k.lower() not in {ck.lower() for ck in config_map}
        }

        category_orders = {
            item["category"]: item["display_order"]
            for item in get_category_definitions()
        }

        schema_by_key: Dict[str, Dict[str, Any]] = {
            key: get_field_definition(key, config_map.get(key, ""))
            for key in all_keys
        }

        items: List[Dict[str, Any]] = []
        for key in all_keys:
            raw_value = config_map.get(key, "")
            field_schema = schema_by_key[key]
            # 如果 .env 中没有该 key，回退到 schema 的 default_value 以保持前后端一致
            if not raw_value and field_schema.get("default_value") is not None:
                raw_value = str(field_schema["default_value"])
            item: Dict[str, Any] = {
                "key": key,
                "value": raw_value,
                "raw_value_exists": bool(raw_value),
                "is_masked": False,
            }
            if include_schema:
                item["schema"] = field_schema
            items.append(item)

        items.sort(
            key=lambda item: (
                category_orders.get(schema_by_key[item["key"]].get("category", "settings"), 999),
                schema_by_key[item["key"]].get("display_order", 9999),
                item["key"],
            )
        )

        return {
            "config_version": self._manager.get_aggregated_config_version(),
            "mask_token": mask_token,
            "items": items,
            "updated_at": self._manager.get_updated_at(),
        }

    def get_config_version(self) -> Dict[str, Any]:
        """Return aggregated config version and update timestamp."""
        return {
            "config_version": self._manager.get_aggregated_config_version(),
            "updated_at": self._manager.get_updated_at(),
        }

    def validate(self, items: Sequence[Dict[str, str]], mask_token: str = "******") -> Dict[str, Any]:
        """Validate submitted items without writing to `.env`."""
        issues = self._collect_issues(items=items, mask_token=mask_token)
        valid = not any(issue["severity"] == "error" for issue in issues)
        return {
            "valid": valid,
            "issues": issues,
        }

    def update(
        self,
        config_version: str,
        items: Sequence[Dict[str, str]],
        mask_token: str = "******",
        reload_now: bool = True,
    ) -> Dict[str, Any]:
        """Validate and persist updates into .env and YAML files, then reload runtime config."""
        current_version = self._manager.get_aggregated_config_version()
        if current_version != config_version:
            raise ConfigConflictError(current_version=current_version)

        issues = self._collect_issues(items=items, mask_token=mask_token)
        errors = [issue for issue in issues if issue["severity"] == "error"]
        if errors:
            raise ConfigValidationError(issues=errors)

        # 分离 .env 更新和 YAML 更新
        env_updates: List[Tuple[str, str]] = []
        yaml_updates: Dict[str, Dict[str, str]] = {}  # yaml_path -> {key: value}
        sensitive_keys: Set[str] = set()

        for item in items:
            key = item["key"]
            value = item["value"]
            field_schema = get_field_definition(key)
            is_sensitive = bool(field_schema.get("is_sensitive", False))

            if ":" in key and not key.isupper():
                # YAML config key (e.g., "config/scorer_config.yaml:scorer.min_score")
                yaml_path, yaml_key = key.split(":", 1)
                if yaml_path not in yaml_updates:
                    yaml_updates[yaml_path] = {}
                yaml_updates[yaml_path][yaml_key] = value
            else:
                # .env config key
                key_upper = key.upper()
                env_updates.append((key_upper, value))
                if is_sensitive:
                    sensitive_keys.add(key_upper)

        # 应用 .env 更新
        updated_keys: List[str] = []
        skipped_masked_keys: List[str] = []
        new_version = current_version

        if env_updates:
            updated_keys, skipped_masked_keys, new_version = self._manager.apply_updates(
                updates=env_updates,
                sensitive_keys=sensitive_keys,
                mask_token=mask_token,
            )

        # 应用 YAML 更新
        yaml_updated_count = 0
        for yaml_path, yaml_flat_map in yaml_updates.items():
            if yaml_path in self._yaml_managers:
                yaml_mgr = self._yaml_managers[yaml_path]
                if yaml_mgr.write_config_map(yaml_flat_map):
                    yaml_updated_count += len(yaml_flat_map)
                    updated_keys.extend([f"{yaml_path}:{k}" for k in yaml_flat_map.keys()])

        warnings: List[str] = []
        reload_triggered = False
        if reload_now and env_updates:
            try:
                Config.reset_instance()
                setup_env(override=True)
                config = Config.get_instance()
                warnings = config.validate()
                reload_triggered = True
            except Exception as exc:  # pragma: no cover - defensive branch
                logger.error("Configuration reload failed: %s", exc, exc_info=True)
                warnings.append("Configuration updated but reload failed")

        new_version = self._manager.get_aggregated_config_version()

        return {
            "success": True,
            "config_version": new_version,
            "applied_count": len(updated_keys),
            "skipped_masked_count": len(skipped_masked_keys),
            "reload_triggered": reload_triggered,
            "updated_keys": updated_keys,
            "warnings": warnings,
        }

    def _collect_issues(self, items: Sequence[Dict[str, str]], mask_token: str) -> List[Dict[str, Any]]:
        """Collect field-level and cross-field validation issues."""
        current_map = self._manager.read_config_map()
        effective_map = dict(current_map)
        issues: List[Dict[str, Any]] = []
        updated_map: Dict[str, str] = {}

        for item in items:
            raw_key = item["key"]
            value = item["value"]

            # 判断是否为 YAML 配置键
            is_yaml_key = ":" in raw_key and not raw_key.isupper()

            if is_yaml_key:
                # YAML key: 只做基本验证，不检查敏感字段和 mask_token
                field_schema = get_field_definition(raw_key, value)
                updated_map[raw_key] = value
                effective_map[raw_key] = value
                issues.extend(self._validate_value(key=raw_key, value=value, field_schema=field_schema))
            else:
                # .env key: 原有流程
                key = raw_key.upper()
                field_schema = get_field_definition(key, value)
                is_sensitive = bool(field_schema.get("is_sensitive", False))

                if is_sensitive and value == mask_token and current_map.get(key):
                    continue

                updated_map[key] = value
                effective_map[key] = value
                issues.extend(self._validate_value(key=key, value=value, field_schema=field_schema))

        issues.extend(self._validate_cross_field(effective_map=effective_map, updated_keys=set(updated_map.keys())))
        return issues

    @staticmethod
    def _validate_value(key: str, value: str, field_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate a single field value against schema metadata."""
        issues: List[Dict[str, Any]] = []
        data_type = field_schema.get("data_type", "string")
        validation = field_schema.get("validation", {}) or {}
        is_required = field_schema.get("is_required", False)

        # Empty values are valid for non-required fields (skip type validation)
        if not value.strip() and not is_required:
            return issues

        if "\n" in value:
            issues.append(
                {
                    "key": key,
                    "code": "invalid_value",
                    "message": "Value cannot contain newline characters",
                    "severity": "error",
                    "expected": "single-line value",
                    "actual": "contains newline",
                }
            )
            return issues

        if data_type == "integer":
            try:
                numeric = int(value)
            except ValueError:
                return [
                    {
                        "key": key,
                        "code": "invalid_type",
                        "message": "Value must be an integer",
                        "severity": "error",
                        "expected": "integer",
                        "actual": value,
                    }
                ]
            issues.extend(SystemConfigService._validate_numeric_range(key, numeric, validation))

        elif data_type == "number":
            try:
                numeric = float(value)
            except ValueError:
                return [
                    {
                        "key": key,
                        "code": "invalid_type",
                        "message": "Value must be a number",
                        "severity": "error",
                        "expected": "number",
                        "actual": value,
                    }
                ]
            issues.extend(SystemConfigService._validate_numeric_range(key, numeric, validation))

        elif data_type == "boolean":
            if value.strip().lower() not in {"true", "false"}:
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_type",
                        "message": "Value must be true or false",
                        "severity": "error",
                        "expected": "true|false",
                        "actual": value,
                    }
                )

        elif data_type == "time":
            pattern = validation.get("pattern") or r"^([01]\d|2[0-3]):[0-5]\d$"
            if not re.match(pattern, value.strip()):
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_format",
                        "message": "Value must be in HH:MM format",
                        "severity": "error",
                        "expected": "HH:MM",
                        "actual": value,
                    }
                )

        if "enum" in validation and value and value not in validation["enum"]:
            issues.append(
                {
                    "key": key,
                    "code": "invalid_enum",
                    "message": "Value is not in allowed options",
                    "severity": "error",
                    "expected": ",".join(validation["enum"]),
                    "actual": value,
                }
            )

        return issues

    @staticmethod
    def _validate_numeric_range(key: str, numeric_value: float, validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        min_value = validation.get("min")
        max_value = validation.get("max")

        if min_value is not None and numeric_value < min_value:
            issues.append(
                {
                    "key": key,
                    "code": "out_of_range",
                    "message": "Value is lower than minimum",
                    "severity": "error",
                    "expected": f">={min_value}",
                    "actual": str(numeric_value),
                }
            )
        if max_value is not None and numeric_value > max_value:
            issues.append(
                {
                    "key": key,
                    "code": "out_of_range",
                    "message": "Value is greater than maximum",
                    "severity": "error",
                    "expected": f"<={max_value}",
                    "actual": str(numeric_value),
                }
            )
        return issues

    @staticmethod
    def _validate_cross_field(effective_map: Dict[str, str], updated_keys: Set[str]) -> List[Dict[str, Any]]:
        """Validate dependencies across multiple keys."""
        issues: List[Dict[str, Any]] = []

        token_value = (effective_map.get("TELEGRAM_BOT_TOKEN") or "").strip()
        chat_id_value = (effective_map.get("TELEGRAM_CHAT_ID") or "").strip()
        if token_value and not chat_id_value and (
            "TELEGRAM_BOT_TOKEN" in updated_keys or "TELEGRAM_CHAT_ID" in updated_keys
        ):
            issues.append(
                {
                    "key": "TELEGRAM_CHAT_ID",
                    "code": "missing_dependency",
                    "message": "TELEGRAM_CHAT_ID is required when TELEGRAM_BOT_TOKEN is set",
                    "severity": "error",
                    "expected": "non-empty TELEGRAM_CHAT_ID",
                    "actual": chat_id_value,
                }
            )

        return issues