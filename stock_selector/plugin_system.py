# -*- coding: utf-8 -*-
"""
插件系统 - 支持策略的插件化加载和热重载
"""

import importlib
import inspect
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """插件信息"""
    name: str
    version: str
    description: str
    author: str
    module_path: Path
    plugin_class: Type
    loaded_at: datetime
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


class PluginSystem:
    """
    插件系统
    
    功能：
    1. 插件自动发现和加载
    2. 插件热重载
    3. 插件生命周期管理
    4. 插件依赖管理
    """
    
    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        """
        初始化插件系统
        
        Args:
            plugin_dirs: 插件目录列表
        """
        self.plugin_dirs = plugin_dirs or []
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugin_instances: Dict[str, Any] = {}
        self._load_hooks: List[Callable[[PluginInfo], None]] = []
        self._unload_hooks: List[Callable[[PluginInfo], None]] = []
    
    def add_plugin_dir(self, plugin_dir: Path) -> None:
        """
        添加插件目录
        
        Args:
            plugin_dir: 插件目录
        """
        if plugin_dir not in self.plugin_dirs:
            self.plugin_dirs.append(plugin_dir)
            logger.info(f"添加插件目录: {plugin_dir}")
    
    def discover_plugins(self) -> List[PluginInfo]:
        """
        自动发现并加载所有插件
        
        Returns:
            加载的插件列表
        """
        loaded_plugins = []
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                logger.warning(f"插件目录不存在: {plugin_dir}")
                continue
            
            # 扫描Python文件
            for py_file in plugin_dir.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                try:
                    plugin_info = self._load_plugin_from_file(py_file)
                    if plugin_info:
                        loaded_plugins.append(plugin_info)
                except Exception as e:
                    logger.error(f"加载插件失败 {py_file}: {e}", exc_info=True)
        
        logger.info(f"插件发现完成，共加载 {len(loaded_plugins)} 个插件")
        return loaded_plugins
    
    def _load_plugin_from_file(self, py_file: Path) -> Optional[PluginInfo]:
        """
        从文件加载插件
        
        Args:
            py_file: Python文件路径
        
        Returns:
            插件信息，加载失败返回None
        """
        # 添加插件目录到sys.path
        plugin_dir = py_file.parent
        if str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))
        
        # 导入模块
        module_name = py_file.stem
        try:
            if module_name in sys.modules:
                # 重新加载模块
                module = importlib.reload(sys.modules[module_name])
            else:
                module = importlib.import_module(module_name)
        except Exception as e:
            logger.error(f"导入模块失败 {module_name}: {e}")
            return None
        
        # 查找插件类
        plugin_class = None
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                hasattr(obj, "is_strategy") and 
                obj.is_strategy):
                plugin_class = obj
                break
        
        if plugin_class is None:
            return None
        
        # 提取插件信息
        plugin_name = getattr(plugin_class, "name", module_name)
        plugin_version = getattr(plugin_class, "version", "1.0.0")
        plugin_description = getattr(plugin_class, "description", "")
        plugin_author = getattr(plugin_class, "author", "unknown")
        plugin_tags = getattr(plugin_class, "tags", [])
        
        # 创建插件信息
        plugin_info = PluginInfo(
            name=plugin_name,
            version=plugin_version,
            description=plugin_description,
            author=plugin_author,
            module_path=py_file,
            plugin_class=plugin_class,
            loaded_at=datetime.now(),
            tags=plugin_tags
        )
        
        # 注册插件
        self._plugins[plugin_name] = plugin_info
        
        # 触发加载钩子
        self._trigger_load_hooks(plugin_info)
        
        logger.info(f"插件加载成功: {plugin_name} v{plugin_version}")
        return plugin_info
    
    def get_plugin(self, name: str) -> Optional[PluginInfo]:
        """
        获取插件信息
        
        Args:
            name: 插件名称
        
        Returns:
            插件信息
        """
        return self._plugins.get(name)
    
    def get_all_plugins(self) -> List[PluginInfo]:
        """
        获取所有插件
        
        Returns:
            插件列表
        """
        return list(self._plugins.values())
    
    def get_enabled_plugins(self) -> List[PluginInfo]:
        """
        获取所有启用的插件
        
        Returns:
            启用的插件列表
        """
        return [p for p in self._plugins.values() if p.enabled]
    
    def get_plugin_instance(self, name: str, *args, **kwargs) -> Optional[Any]:
        """
        获取插件实例（单例模式）
        
        Args:
            name: 插件名称
            *args: 实例化参数
            **kwargs: 实例化关键字参数
        
        Returns:
            插件实例
        """
        plugin_info = self._plugins.get(name)
        if not plugin_info or not plugin_info.enabled:
            return None
        
        if name not in self._plugin_instances:
            try:
                self._plugin_instances[name] = plugin_info.plugin_class(*args, **kwargs)
                logger.info(f"插件实例化成功: {name}")
            except Exception as e:
                logger.error(f"插件实例化失败 {name}: {e}", exc_info=True)
                return None
        
        return self._plugin_instances[name]
    
    def enable_plugin(self, name: str) -> bool:
        """
        启用插件
        
        Args:
            name: 插件名称
        
        Returns:
            是否成功
        """
        if name in self._plugins:
            self._plugins[name].enabled = True
            logger.info(f"插件已启用: {name}")
            return True
        return False
    
    def disable_plugin(self, name: str) -> bool:
        """
        禁用插件
        
        Args:
            name: 插件名称
        
        Returns:
            是否成功
        """
        if name in self._plugins:
            self._plugins[name].enabled = False
            # 清除实例
            if name in self._plugin_instances:
                del self._plugin_instances[name]
            logger.info(f"插件已禁用: {name}")
            return True
        return False
    
    def reload_plugin(self, name: str) -> Optional[PluginInfo]:
        """
        热重载插件
        
        Args:
            name: 插件名称
        
        Returns:
            重新加载的插件信息
        """
        plugin_info = self._plugins.get(name)
        if not plugin_info:
            logger.warning(f"插件不存在: {name}")
            return None
        
        # 触发卸载钩子
        self._trigger_unload_hooks(plugin_info)
        
        # 清除实例
        if name in self._plugin_instances:
            del self._plugin_instances[name]
        
        # 重新加载
        return self._load_plugin_from_file(plugin_info.module_path)
    
    def unload_plugin(self, name: str) -> bool:
        """
        卸载插件
        
        Args:
            name: 插件名称
        
        Returns:
            是否成功
        """
        if name not in self._plugins:
            return False
        
        plugin_info = self._plugins[name]
        
        # 触发卸载钩子
        self._trigger_unload_hooks(plugin_info)
        
        # 清除实例
        if name in self._plugin_instances:
            del self._plugin_instances[name]
        
        # 删除插件
        del self._plugins[name]
        
        logger.info(f"插件已卸载: {name}")
        return True
    
    def register_load_hook(self, hook: Callable[[PluginInfo], None]) -> None:
        """
        注册插件加载钩子
        
        Args:
            hook: 钩子函数
        """
        self._load_hooks.append(hook)
    
    def register_unload_hook(self, hook: Callable[[PluginInfo], None]) -> None:
        """
        注册插件卸载钩子
        
        Args:
            hook: 钩子函数
        """
        self._unload_hooks.append(hook)
    
    def _trigger_load_hooks(self, plugin_info: PluginInfo) -> None:
        """触发插件加载钩子"""
        for hook in self._load_hooks:
            try:
                hook(copy.deepcopy(plugin_info))
            except Exception as e:
                logger.error(f"插件加载钩子执行失败: {e}", exc_info=True)
    
    def _trigger_unload_hooks(self, plugin_info: PluginInfo) -> None:
        """触发插件卸载钩子"""
        for hook in self._unload_hooks:
            try:
                hook(copy.deepcopy(plugin_info))
            except Exception as e:
                logger.error(f"插件卸载钩子执行失败: {e}", exc_info=True)


# 全局插件系统实例
_plugin_system: Optional[PluginSystem] = None


def get_plugin_system(plugin_dirs: Optional[List[Path]] = None) -> PluginSystem:
    """
    获取全局插件系统实例
    
    Args:
        plugin_dirs: 插件目录列表
    
    Returns:
        插件系统实例
    """
    global _plugin_system
    if _plugin_system is None:
        _plugin_system = PluginSystem(plugin_dirs)
    return _plugin_system
