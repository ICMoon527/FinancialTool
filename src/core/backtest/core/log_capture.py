# -*- coding: utf-8 -*-
"""
日志捕获器。

用于捕获回测过程中的日志并实时推送给前端。
"""

import logging
from datetime import datetime
from queue import Queue
from typing import List, Optional


class LogCaptureHandler(logging.Handler):
    """日志捕获处理器"""

    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            self.queue.put(f"[{timestamp}] {msg}")
        except Exception:
            self.handleError(record)


class LogCapturer:
    """日志捕获器"""

    _instance: Optional["LogCapturer"] = None
    _queue: Optional[Queue] = None
    _handler: Optional[LogCaptureHandler] = None
    _logger_names: List[str] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._queue = Queue()
        return cls._instance

    def start_capture(self, logger_names: Optional[List[str]] = None) -> None:
        """
        开始捕获日志。

        Args:
            logger_names: 要捕获的日志器名称列表，如果为None则捕获所有
        """
        if self._handler is not None:
            self.stop_capture()

        self._handler = LogCaptureHandler(self._queue)
        formatter = logging.Formatter("%(message)s")
        self._handler.setFormatter(formatter)

        if logger_names:
            self._logger_names = logger_names
            for name in logger_names:
                logger = logging.getLogger(name)
                logger.addHandler(self._handler)
        else:
            # 捕获根日志器
            self._logger_names = [""]
            root_logger = logging.getLogger()
            root_logger.addHandler(self._handler)

    def stop_capture(self) -> None:
        """停止捕获日志"""
        if self._handler is None:
            return

        for name in self._logger_names:
            logger = logging.getLogger(name)
            logger.removeHandler(self._handler)

        self._handler = None
        self._logger_names = []

    def get_logs(self, timeout: float = 0.1) -> Optional[str]:
        """
        获取捕获的日志。

        Args:
            timeout: 等待超时时间（秒）

        Returns:
            日志消息，如果队列为空则返回None
        """
        if self._queue is None:
            return None

        try:
            return self._queue.get(timeout=timeout)
        except Exception:
            return None

    def clear(self) -> None:
        """清空日志队列"""
        if self._queue is not None:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except Exception:
                    break


def get_log_capturer() -> LogCapturer:
    """获取日志捕获器单例"""
    return LogCapturer()
