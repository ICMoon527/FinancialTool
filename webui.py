# -*- coding: utf-8 -*-
"""
===================================
WebUI 启动网页可视化脚本
===================================

用于启动 Web 服务界面。
直接运行 `python webui.py` 将启动 Web 后端服务。

等效命令：
    python main.py --webui-only

Usage:
  python webui.py
  WEBUI_HOST=0.0.0.0 WEBUI_PORT=8000 python webui.py
"""

from __future__ import annotations

import os

# 必须在任何 numpy 导入之前设置：
# 禁用 Intel Fortran 运行时（libifcoremd.dll）注册控制台 Ctrl+C 处理器。
# 否则它会在评估/训练加载数据后抢先拦截 Ctrl+C，打印 forrtl error (200) 并挂起进程，
# 导致 Python 侧的信号处理与看门狗完全失效。
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

import logging
import warnings

# 抑制 newspaper3k 等第三方库在 Python 3.12+ 中的无效转义序列警告
warnings.filterwarnings("ignore", category=SyntaxWarning, module="newspaper")

logger = logging.getLogger(__name__)


def main() -> int:
    """
    启动 Web 服务
    """
    # 兼容旧版环境变量名
    host = os.getenv("WEBUI_HOST", os.getenv("API_HOST", "127.0.0.1"))
    port = int(os.getenv("WEBUI_PORT", os.getenv("API_PORT", "8000")))

    print(f"正在启动 Web 服务: http://{host}:{port}")
    print(f"API 文档: http://{host}:{port}/docs")
    print()

    try:
        import contextlib
        import signal
        import threading
        import time

        import uvicorn
        from src.config import setup_env
        from src.logging_config import setup_logging

        setup_env()
        setup_logging(log_prefix="web_server")

        class NoSignalServer(uvicorn.Server):
            """禁用 uvicorn 自带信号处理，SIGINT 统一由本入口在主线程接管

            背景：评估/训练任务中 numpy 会加载 Intel Fortran 运行时（libifcoremd.dll），
            其自带的控制台 Ctrl+C 处理器会与 uvicorn 的优雅关闭冲突，导致进程卡死。
            """

            @contextlib.contextmanager
            def capture_signals(self):
                yield

        server = NoSignalServer(
            uvicorn.Config(
                "api.app:app",
                host=host,
                port=port,
                log_level="info",
                timeout_graceful_shutdown=3,
            )
        )

        _interrupted = []

        def _watchdog() -> None:
            # 看门狗：优雅关闭超时则强制终止，避免 Fortran 运行时挂起进程
            time.sleep(5)
            os._exit(1)

        def _handle_sigint(signum, frame) -> None:
            # 信号处理器仅在主线程注册和执行
            if not _interrupted:
                _interrupted.append(True)
                print("\n收到 Ctrl+C，正在优雅关闭（再按一次立即退出）...")
                server.should_exit = True
                threading.Thread(target=_watchdog, daemon=True).start()
            else:
                os._exit(1)

        signal.signal(signal.SIGINT, _handle_sigint)

        server.run()
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
