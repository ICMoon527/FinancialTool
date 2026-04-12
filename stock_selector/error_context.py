# -*- coding: utf-8 -*-
"""
错误上下文模块 - 增强错误信息的上下文和追踪
"""

import logging
import uuid
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ErrorSnapshot:
    """错误快照"""
    timestamp: datetime
    error_id: str
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class ErrorContextManager:
    """
    错误上下文管理器
    
    功能：
    1. 错误唯一追踪ID
    2. 完整的错误上下文记录
    3. 错误快照功能
    4. 错误关联分析
    5. 错误报告导出
    """
    
    def __init__(self, snapshot_dir: Optional[Path] = None):
        """
        初始化错误上下文管理器
        
        Args:
            snapshot_dir: 错误快照保存目录
        """
        self.snapshot_dir = snapshot_dir or Path("error_snapshots")
        self.snapshot_dir.mkdir(exist_ok=True)
        self._snapshots: List[ErrorSnapshot] = []
        self._max_snapshots = 1000
        self._context_stack: List[Dict[str, Any]] = []
        self._error_hooks: List[Callable[[ErrorSnapshot], None]] = []
    
    def push_context(self, context: Dict[str, Any]) -> None:
        """
        推送上下文
        
        Args:
            context: 上下文数据
        """
        self._context_stack.append(dict(context))
    
    def pop_context(self) -> Optional[Dict[str, Any]]:
        """
        弹出上下文
        
        Returns:
            弹出的上下文数据
        """
        if self._context_stack:
            return self._context_stack.pop()
        return None
    
    def get_current_context(self) -> Dict[str, Any]:
        """
        获取当前合并的上下文
        
        Returns:
            合并后的上下文数据
        """
        merged_context = {}
        for ctx in self._context_stack:
            merged_context.update(ctx)
        return merged_context
    
    def capture_error(
        self,
        error: Exception,
        additional_context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> ErrorSnapshot:
        """
        捕获错误并创建快照
        
        Args:
            error: 异常对象
            additional_context: 额外的上下文
            tags: 错误标签
        
        Returns:
            错误快照
        """
        error_id = str(uuid.uuid4())
        
        # 合并上下文
        context = self.get_current_context()
        if additional_context:
            context.update(additional_context)
        
        # 创建错误快照
        snapshot = ErrorSnapshot(
            timestamp=datetime.now(),
            error_id=error_id,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context,
            tags=tags or []
        )
        
        # 保存快照
        self._snapshots.append(snapshot)
        
        # 限制快照数量
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]
        
        # 保存到文件
        self._save_snapshot(snapshot)
        
        # 触发错误钩子
        self._trigger_error_hooks(snapshot)
        
        logger.error(
            f"[错误ID: {error_id}] {snapshot.error_type}: {snapshot.error_message}",
            extra={'error_id': error_id}
        )
        
        return snapshot
    
    def _save_snapshot(self, snapshot: ErrorSnapshot) -> None:
        """
        保存错误快照到文件
        
        Args:
            snapshot: 错误快照
        """
        try:
            timestamp_str = snapshot.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"error_{timestamp_str}_{snapshot.error_id[:8]}.json"
            filepath = self.snapshot_dir / filename
            
            snapshot_dict = asdict(snapshot)
            # 转换datetime为字符串
            snapshot_dict['timestamp'] = snapshot.timestamp.isoformat()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot_dict, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"错误快照已保存: {filepath}")
        except Exception as e:
            logger.error(f"保存错误快照失败: {e}")
    
    def get_snapshot(self, error_id: str) -> Optional[ErrorSnapshot]:
        """
        根据错误ID获取快照
        
        Args:
            error_id: 错误ID
        
        Returns:
            错误快照
        """
        for snapshot in self._snapshots:
            if snapshot.error_id == error_id:
                return snapshot
        return None
    
    def get_all_snapshots(
        self,
        error_type: Optional[str] = None,
        tag: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ErrorSnapshot]:
        """
        获取错误快照列表，支持筛选
        
        Args:
            error_type: 错误类型筛选
            tag: 标签筛选
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制
        
        Returns:
            错误快照列表
        """
        snapshots = self._snapshots
        
        # 筛选
        if error_type:
            snapshots = [s for s in snapshots if s.error_type == error_type]
        if tag:
            snapshots = [s for s in snapshots if tag in s.tags]
        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]
        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]
        
        # 按时间倒序并限制数量
        snapshots = sorted(snapshots, key=lambda x: x.timestamp, reverse=True)
        return snapshots[:limit]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        获取错误统计信息
        
        Returns:
            统计信息
        """
        if not self._snapshots:
            return {'total': 0}
        
        error_type_counts = {}
        tag_counts = {}
        time_range = {
            'first': self._snapshots[0].timestamp,
            'last': self._snapshots[-1].timestamp
        }
        
        for snapshot in self._snapshots:
            error_type_counts[snapshot.error_type] = error_type_counts.get(snapshot.error_type, 0) + 1
            for tag in snapshot.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            'total': len(self._snapshots),
            'error_types': error_type_counts,
            'tags': tag_counts,
            'time_range': time_range
        }
    
    def export_snapshots(self, filepath: Path, **kwargs) -> int:
        """
        导出错误快照
        
        Args:
            filepath: 导出文件路径
            **kwargs: 筛选参数，同 get_all_snapshots
        
        Returns:
            导出的快照数量
        """
        snapshots = self.get_all_snapshots(**kwargs)
        
        export_data = []
        for snapshot in snapshots:
            snapshot_dict = asdict(snapshot)
            snapshot_dict['timestamp'] = snapshot.timestamp.isoformat()
            export_data.append(snapshot_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已导出 {len(snapshots)} 个错误快照到: {filepath}")
        return len(snapshots)
    
    def clear_snapshots(self, before: Optional[datetime] = None) -> int:
        """
        清除错误快照
        
        Args:
            before: 清除此时间之前的快照，None表示清除所有
        
        Returns:
            清除的快照数量
        """
        original_count = len(self._snapshots)
        
        if before:
            self._snapshots = [s for s in self._snapshots if s.timestamp > before]
        else:
            self._snapshots = []
        
        cleared_count = original_count - len(self._snapshots)
        logger.info(f"已清除 {cleared_count} 个错误快照")
        return cleared_count
    
    def register_error_hook(self, hook: Callable[[ErrorSnapshot], None]) -> None:
        """
        注册错误钩子
        
        Args:
            hook: 钩子函数
        """
        self._error_hooks.append(hook)
    
    def unregister_error_hook(self, hook: Callable[[ErrorSnapshot], None]) -> None:
        """
        注销错误钩子
        
        Args:
            hook: 钩子函数
        """
        if hook in self._error_hooks:
            self._error_hooks.remove(hook)
    
    def _trigger_error_hooks(self, snapshot: ErrorSnapshot) -> None:
        """触发错误钩子"""
        for hook in self._error_hooks:
            try:
                hook(snapshot)
            except Exception as e:
                logger.error(f"错误钩子执行失败: {e}", exc_info=True)


# 全局错误上下文管理器实例
_error_context_manager: Optional[ErrorContextManager] = None


def get_error_context_manager(snapshot_dir: Optional[Path] = None) -> ErrorContextManager:
    """
    获取全局错误上下文管理器实例
    
    Args:
        snapshot_dir: 错误快照保存目录
    
    Returns:
        错误上下文管理器实例
    """
    global _error_context_manager
    if _error_context_manager is None:
        _error_context_manager = ErrorContextManager(snapshot_dir)
    return _error_context_manager
