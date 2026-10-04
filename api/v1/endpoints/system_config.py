# -*- coding: utf-8 -*-
"""System configuration endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_system_config_service
from api.v1.schemas.common import ErrorResponse
from api.v1.schemas.system_config import (
    PickFileResponse,
    SystemConfigConflictResponse,
    SystemConfigResponse,
    SystemConfigSchemaResponse,
    SystemConfigValidationErrorResponse,
    SystemConfigVersionResponse,
    UpdateSystemConfigRequest,
    UpdateSystemConfigResponse,
    ValidateSystemConfigRequest,
    ValidateSystemConfigResponse,
)
from src.services.system_config_service import ConfigConflictError, ConfigValidationError, SystemConfigService

logger = logging.getLogger(__name__)

router = APIRouter()


def _pick_file_windows(title: str, initial_dir: str = "", file_filter: str = "SQLite 数据库 (*.db)\0*.db\0所有文件 (*.*)\0*.*\0\0") -> str:
    """使用 Windows 原生打开文件对话框选择文件，返回选中文件的绝对路径；取消时返回空字符串。"""
    import ctypes
    from ctypes import wintypes

    # 定义 Windows OPENFILENAME 结构体
    class OPENFILENAME(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", wintypes.LPCWSTR),
        ]

    file_buffer = ctypes.create_unicode_buffer(1024)
    ofn = OPENFILENAME()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
    ofn.lpstrFilter = file_filter
    ofn.lpstrFile = file_buffer
    ofn.nMaxFile = 1024
    ofn.lpstrTitle = title
    if initial_dir:
        ofn.lpstrInitialDir = initial_dir
    ofn.Flags = 0x00001000 | 0x00000200 | 0x00000400  # OFN_PATHMUSTEXIST | OFN_HIDEREADONLY | OFN_FILEMUSTEXIST

    get_open_file_name = ctypes.windll.comdlg32.GetOpenFileNameW
    get_open_file_name.argtypes = [ctypes.POINTER(OPENFILENAME)]
    get_open_file_name.restype = wintypes.BOOL

    if get_open_file_name(ctypes.byref(ofn)):
        return file_buffer.value
    return ""


def _pick_file(title: str, initial_dir: str = "") -> str:
    """弹出本机文件选择窗口，返回选中文件的绝对路径；取消时返回空字符串。"""
    # 优先使用 Windows 原生对话框（与操作系统文件管理器体验一致）
    try:
        return _pick_file_windows(title, initial_dir)
    except Exception as win_err:  # 非 Windows 环境或调用失败时回退到 tkinter
        logger.debug("Windows 原生文件对话框不可用，回退 tkinter: %s", win_err)
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                path = filedialog.askopenfilename(
                    parent=root,
                    title=title,
                    initialdir=initial_dir or None,
                    filetypes=[("SQLite 数据库", "*.db"), ("所有文件", "*.*")],
                )
                return path or ""
            finally:
                root.destroy()
        except Exception as tk_err:
            raise RuntimeError(f"无法打开文件选择窗口（Windows 原生失败: {win_err}；tkinter 失败: {tk_err}）") from tk_err


@router.get(
    "/config",
    response_model=SystemConfigResponse,
    responses={
        200: {"description": "Configuration loaded"},
        401: {"description": "Unauthorized", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Get system configuration",
    description="Read current configuration from .env and return raw values.",
)
def get_system_config(
    include_schema: bool = Query(True, description="Whether to include schema metadata"),
    service: SystemConfigService = Depends(get_system_config_service),
) -> SystemConfigResponse:
    """Load and return current system configuration."""
    try:
        payload = service.get_config(include_schema=include_schema)
        return SystemConfigResponse.model_validate(payload)
    except Exception as exc:
        logger.error("Failed to load system configuration: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "Failed to load system configuration",
            },
        )


@router.put(
    "/config",
    response_model=UpdateSystemConfigResponse,
    responses={
        200: {"description": "Configuration updated"},
        400: {"description": "Validation failed", "model": SystemConfigValidationErrorResponse},
        409: {"description": "Version conflict", "model": SystemConfigConflictResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Update system configuration",
    description="Update key-value pairs in .env. Mask token preserves existing secret values.",
)
def update_system_config(
    request: UpdateSystemConfigRequest,
    service: SystemConfigService = Depends(get_system_config_service),
) -> UpdateSystemConfigResponse:
    """Validate and persist system configuration updates."""
    try:
        payload = service.update(
            config_version=request.config_version,
            items=[item.model_dump() for item in request.items],
            mask_token=request.mask_token,
            reload_now=request.reload_now,
        )
        # 若本次更新涉及 RL_* 配置，通知 RLService 就地重载（保留运行中任务）
        updated_keys = [str(key) for key in payload.get("updated_keys", [])]
        if any(key.upper().startswith("RL_") for key in updated_keys):
            try:
                from api.v1.endpoints.rl import reload_rl_service_config

                reload_rl_service_config()
            except Exception as rl_exc:  # 重载失败不应影响配置保存结果
                logger.warning("RL 配置重载失败: %s", rl_exc, exc_info=True)
        return UpdateSystemConfigResponse.model_validate(payload)
    except ConfigValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_failed",
                "message": "System configuration validation failed",
                "issues": exc.issues,
            },
        )
    except ConfigConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "config_version_conflict",
                "message": "Configuration has changed, please reload and retry",
                "current_config_version": exc.current_version,
            },
        )
    except Exception as exc:
        logger.error("Failed to update system configuration: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "Failed to update system configuration",
            },
        )


@router.post(
    "/config/validate",
    response_model=ValidateSystemConfigResponse,
    responses={
        200: {"description": "Validation completed"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Validate system configuration",
    description="Validate submitted configuration values without writing to .env.",
)
def validate_system_config(
    request: ValidateSystemConfigRequest,
    service: SystemConfigService = Depends(get_system_config_service),
) -> ValidateSystemConfigResponse:
    """Run pre-save validation only."""
    try:
        payload = service.validate(items=[item.model_dump() for item in request.items])
        return ValidateSystemConfigResponse.model_validate(payload)
    except Exception as exc:
        logger.error("Failed to validate system configuration: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "Failed to validate system configuration",
            },
        )


@router.get(
    "/config/schema",
    response_model=SystemConfigSchemaResponse,
    responses={
        200: {"description": "Schema loaded"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Get system configuration schema",
    description="Return categorized field metadata used for dynamic settings form rendering.",
)
def get_system_config_schema(
    service: SystemConfigService = Depends(get_system_config_service),
) -> SystemConfigSchemaResponse:
    """Return schema metadata for system configuration fields."""
    try:
        payload = service.get_schema()
        return SystemConfigSchemaResponse.model_validate(payload)
    except Exception as exc:
        logger.error("Failed to load system configuration schema: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "Failed to load system configuration schema",
            },
        )


@router.get(
    "/config/version",
    response_model=SystemConfigVersionResponse,
    responses={
        200: {"description": "Config version retrieved"},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Get config version",
    description="Return aggregated config version hash for polling-based change detection.",
)
def get_system_config_version(
    service: SystemConfigService = Depends(get_system_config_service),
) -> SystemConfigVersionResponse:
    """Return current config version for frontend polling."""
    try:
        payload = service.get_config_version()
        return SystemConfigVersionResponse.model_validate(payload)
    except Exception as exc:
        logger.error("Failed to get config version: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "Failed to get config version",
            },
        )


@router.post(
    "/config/pick-file",
    response_model=PickFileResponse,
    responses={
        200: {"description": "File picked or canceled"},
        500: {"description": "Failed to open file dialog", "model": ErrorResponse},
    },
    summary="Pick a file via native dialog",
    description="Open a native file picker on the server machine and return the selected absolute path.",
)
def pick_file(
    service: SystemConfigService = Depends(get_system_config_service),
) -> PickFileResponse:
    """在后端本机弹出系统原生文件选择窗口，返回选中的文件绝对路径。"""
    try:
        # 初始目录：优先使用当前 DATABASE_PATH 所在的目录，便于直接找到现有数据库
        initial_dir = ""
        try:
            current_path = next(
                (item["value"] for item in service.get_config(include_schema=False).get("items", []) if item["key"] == "DATABASE_PATH"),
                "",
            )
            if current_path:
                import os

                candidate = os.path.dirname(current_path)
                if candidate and os.path.isdir(candidate):
                    initial_dir = candidate
        except Exception as dir_exc:
            logger.debug("获取数据库初始目录失败，忽略: %s", dir_exc)

        selected = _pick_file(title="选择数据库文件", initial_dir=initial_dir)
        if not selected:
            return PickFileResponse(success=False, path=None, message="用户取消选择")
        return PickFileResponse(success=True, path=selected)
    except Exception as exc:
        logger.error("Failed to open file dialog: %s", exc, exc_info=True)
        return PickFileResponse(
            success=False,
            path=None,
            message=f"无法打开文件选择窗口: {exc}",
        )
