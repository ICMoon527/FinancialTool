# -*- coding: utf-8 -*-
"""
Watchdog Notifier Module

Integration with project's existing notification system.
"""

import logging
import tkinter as tk
from tkinter import messagebox
from typing import List, Optional

from watchdog.base import WatchdogAlert

logger = logging.getLogger(__name__)


class WatchdogNotifier:
    """
    Watchdog Notifier - sends alerts via project's notification system.
    """

    def __init__(self, enable_notifications: bool = True):
        """
        Initialize the notifier.

        Args:
            enable_notifications: Whether to enable notifications
        """
        self.enable_notifications = enable_notifications
        self._notification_service = None
        self._initialize_notification_service()

    def _initialize_notification_service(self) -> None:
        """Initialize the project's notification service."""
        if not self.enable_notifications:
            return

        try:
            from src.notification import NotificationService
            self._notification_service = NotificationService()
            logger.info("Notification service initialized successfully")
        except ImportError as e:
            logger.warning(f"Notification service not available: {e}")
            self._notification_service = None
        except Exception as e:
            logger.error(f"Failed to initialize notification service: {e}")
            self._notification_service = None

    def is_available(self) -> bool:
        """Check if notification service is available."""
        if not self.enable_notifications:
            return False
        return self._notification_service is not None and self._notification_service.is_available()

    def format_alert_message(self, alert: WatchdogAlert) -> str:
        """
        Format an alert as a Markdown message.

        Args:
            alert: WatchdogAlert to format

        Returns:
            Formatted Markdown string
        """
        level_emojis = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
        }
        emoji = level_emojis.get(alert.alert_level.value, "📢")

        lines = [
            f"# {emoji} 盯盘预警 - {alert.stock_name}({alert.stock_code})",
            "",
            f"> 触发时间：{alert.trigger_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            f"## 📊 预警信息",
            "",
            f"**策略名称**: {alert.strategy_name}",
            "",
            f"**预警级别**: {alert.alert_level.value.upper()}",
            "",
            f"**预警消息**: {alert.message}",
            "",
        ]

        if alert.triggered_conditions:
            lines.extend([
                "## 🔧 触发条件",
                "",
            ])
            for cond in alert.triggered_conditions:
                lines.append(f"- {cond.condition_type.value}: {cond.description}")
            lines.append("")

        if alert.current_data:
            lines.extend([
                "## 📈 当前行情",
                "",
            ])
            for key, value in alert.current_data.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        if alert.decision:
            lines.extend([
                "## 🎯 决策建议",
                "",
                f"**操作方向**: {alert.decision.action.value}",
                "",
                f"**建议理由**: {alert.decision.reason}",
                "",
            ])
            if alert.decision.target_price:
                lines.append(f"**目标价位**: {alert.decision.target_price}")
            if alert.decision.stop_loss:
                lines.append(f"**止损价位**: {alert.decision.stop_loss}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "*盯盘助手自动预警*",
        ])

        return "\n".join(lines)

    def send_alert(self, alert: WatchdogAlert) -> bool:
        """
        Send an alert via all configured channels.

        Args:
            alert: WatchdogAlert to send

        Returns:
            True if sent successfully
        """
        if not self.enable_notifications:
            logger.debug("Notifications disabled, not sending alert")
            return False

        try:
            # 弹出对话框通知（简单设置）
            self._show_popup_alert(alert)
            
            # 同时尝试使用项目的通知系统
            success = False
            if self._notification_service and self.is_available():
                message = self.format_alert_message(alert)
                if hasattr(self._notification_service, '_has_context_channel') and self._notification_service._has_context_channel():
                    success = self._notification_service.send_to_context(message)

                if not success and self._notification_service.get_available_channels():
                    if hasattr(self._notification_service, 'send_report'):
                        self._notification_service.send_report(message)
                        success = True

            logger.info(f"Alert sent for {alert.stock_code}")
            return True

        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
            
    def _show_popup_alert(self, alert: WatchdogAlert) -> None:
        """
        Show a popup dialog for the alert.

        Args:
            alert: WatchdogAlert to display
        """
        try:
            # 创建主窗口
            root = tk.Tk()
            root.withdraw()  # 隐藏主窗口
            
            # 构建消息内容
            title = f"盯盘预警 - {alert.stock_name}({alert.stock_code})"
            
            message_lines = [
                f"触发时间: {alert.trigger_time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"策略名称: {alert.strategy_name}",
                f"预警级别: {alert.alert_level.value.upper()}",
                f"预警消息: {alert.message}",
                ""
            ]
            
            if alert.triggered_conditions:
                message_lines.append("触发条件:")
                for cond in alert.triggered_conditions:
                    message_lines.append(f"  - {cond.description}")
                message_lines.append("")
            
            if alert.decision:
                message_lines.append("决策建议:")
                message_lines.append(f"  操作方向: {alert.decision.action.value}")
                message_lines.append(f"  建议理由: {alert.decision.reason}")
                if alert.decision.target_price:
                    message_lines.append(f"  目标价位: {alert.decision.target_price}")
                if alert.decision.stop_loss:
                    message_lines.append(f"  止损价位: {alert.decision.stop_loss}")
            
            message = "\n".join(message_lines)
            
            # 根据预警级别选择图标
            if alert.alert_level.value == "critical":
                messagebox.showerror(title, message)
            elif alert.alert_level.value == "warning":
                messagebox.showwarning(title, message)
            else:
                messagebox.showinfo(title, message)
            
            root.destroy()
        except Exception as e:
            logger.error(f"Error showing popup alert: {e}")

    def send_alerts(self, alerts: List[WatchdogAlert]) -> int:
        """
        Send multiple alerts.

        Args:
            alerts: List of WatchdogAlert to send

        Returns:
            Number of alerts sent successfully
        """
        count = 0
        for alert in alerts:
            if self.send_alert(alert):
                count += 1
        return count
