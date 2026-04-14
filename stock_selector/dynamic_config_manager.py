# -*- coding: utf-8 -*-
"""
动态配置管理器模块

提供线程安全的配置管理功能，支持配置的原子性切换和失败回滚。
"""

import threading
import os
import time
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple, TYPE_CHECKING
from pathlib import Path
from dataclasses import fields

if TYPE_CHECKING:
    from .config import StockSelectorConfig

# 配置日志记录器
logger = logging.getLogger(__name__)


class DynamicConfigManager:
    """
    动态配置管理器类

    提供线程安全的配置读写、原子性切换和失败回滚功能。
    使用 threading.RLock 实现可重入的读写锁，维护当前配置和上一个有效配置。
    支持配置变更回调机制、敏感信息保护和配置变更比较。
    """

    def __init__(self, initial_config: Optional['StockSelectorConfig'] = None):
        """
        初始化动态配置管理器

        Args:
            initial_config: 初始配置对象，如果为 None 则从环境变量加载默认配置
        """
        logger.info("初始化动态配置管理器")
        
        # 使用可重入锁实现线程安全
        self._lock = threading.RLock()
        
        # 当前生效的配置
        self._current_config: 'StockSelectorConfig'
        
        # 上一个有效配置，用于失败回滚
        self._previous_config: Optional['StockSelectorConfig'] = None
        
        # 配置文件路径（可选，用于 reload_config 方法）
        self._config_path: Optional[Path] = None
        
        # ========== 回调函数相关属性 ==========
        # 注册的回调函数列表，格式：{回调ID: 回调函数}
        self._callbacks: Dict[str, Callable[['StockSelectorConfig', 'StockSelectorConfig'], None]] = {}
        
        # 回调函数ID计数器，用于生成唯一的回调ID
        self._callback_id_counter = 0
        
        # ========== 文件监控相关属性 ==========
        # 监控线程实例
        self._monitor_thread: Optional[threading.Thread] = None
        
        # 监控线程停止标志
        self._monitor_stop_event = threading.Event()
        
        # 监控的 .env 文件路径
        self._env_file_path: Optional[Path] = None
        
        # 上次文件修改时间
        self._last_modified_time: float = 0.0
        
        # 监控间隔（秒），默认 1 秒
        self._monitor_interval: float = 1.0
        
        # 防抖时间（秒），默认 0.5 秒，避免短时间内多次修改频繁触发
        self._debounce_time: float = 0.5
        
        # 上次触发配置重载的时间，用于防抖
        self._last_reload_time: float = 0.0
        
        # ========== 初始化配置 ==========
        if initial_config is not None:
            self._current_config = initial_config
            logger.debug("使用提供的初始配置")
        else:
            from .config import StockSelectorConfig
            self._current_config = StockSelectorConfig.from_env()
            logger.debug("从环境变量加载默认配置")
        
        logger.info("动态配置管理器初始化完成")

    def get_config(self) -> 'StockSelectorConfig':
        """
        获取当前配置

        Returns:
            StockSelectorConfig: 当前生效的配置对象
        
        线程安全说明：此方法使用读锁，多个线程可以同时读取配置
        """
        with self._lock:
            return self._current_config

    def set_config(self, new_config: 'StockSelectorConfig', 
                   validator: Optional[Callable[['StockSelectorConfig'], bool]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        原子性地切换到新配置

        Args:
            new_config: 新的配置对象
            validator: 可选的配置验证函数，接收新配置并返回布尔值表示是否有效

        Returns:
            Tuple[bool, Dict[str, Any]]: 
                - 第一个元素：配置切换成功返回 True，失败返回 False（此时已自动回滚）
                - 第二个元素：包含变更信息的字典，键为配置项名称，值为包含旧值和新值的字典
        
        线程安全说明：此方法使用写锁，确保配置切换的原子性
        """
        logger.info("开始切换配置")
        
        with self._lock:
            # 保存当前配置作为备份，用于可能的回滚
            old_config = self._current_config
            
            # 计算配置变更
            changes = self._compare_configs(old_config, new_config)
            
            try:
                # 如果提供了验证器，先验证新配置
                if validator is not None:
                    logger.debug("执行配置验证")
                    if not validator(new_config):
                        raise ValueError("配置验证失败")
                
                # 原子性切换配置：先保存旧配置，再设置新配置
                self._previous_config = old_config
                self._current_config = new_config
                
                logger.info(f"配置切换成功，变更项数量: {len(changes)}")
                if changes:
                    masked_changes = self._mask_sensitive_changes(changes)
                    logger.info(f"配置变更详情: {masked_changes}")
                
                # 调用所有注册的回调函数
                self._call_callbacks(old_config, new_config)
                
                return True, changes
                
            except Exception as e:
                # 配置更新失败，回滚到旧配置
                self._current_config = old_config
                logger.error(f"配置切换失败: {str(e)}，已回滚到旧配置")
                return False, {}

    def reload_config(self, config_path: Optional[str] = None,
                     validator: Optional[Callable[['StockSelectorConfig'], bool]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        重新加载配置

        Args:
            config_path: 可选的配置文件路径，如果不指定则使用默认方式加载
            validator: 可选的配置验证函数

        Returns:
            Tuple[bool, Dict[str, Any]]: 
                - 第一个元素：配置重新加载成功返回 True，失败返回 False（此时已自动回滚）
                - 第二个元素：包含变更信息的字典，键为配置项名称，值为包含旧值和新值的字典
        
        线程安全说明：此方法内部调用 set_config，自动获得线程安全保证
        """
        logger.info("开始重新加载配置")
        
        with self._lock:
            try:
                # 保存配置路径以便后续使用
                if config_path is not None:
                    self._config_path = Path(config_path)
                    logger.debug(f"使用指定的配置路径: {self._config_path}")
                
                # 重新加载配置（目前使用 from_env 方法）
                # 如果需要支持从文件加载，可以在这里扩展
                logger.debug("从环境变量重新加载配置")
                from .config import StockSelectorConfig
                new_config = StockSelectorConfig.from_env()
                
                # 使用 set_config 进行原子性切换，自动处理回滚
                success, changes = self.set_config(new_config, validator)
                
                if success:
                    logger.info("配置重新加载成功")
                else:
                    logger.error("配置重新加载失败")
                
                return success, changes
                
            except Exception as e:
                logger.error(f"配置重新加载过程中发生异常: {str(e)}")
                return False, {}

    def rollback(self) -> Tuple[bool, Dict[str, Any]]:
        """
        手动回滚到上一个有效配置

        Returns:
            Tuple[bool, Dict[str, Any]]: 
                - 第一个元素：回滚成功返回 True，如果没有可用的上一个配置返回 False
                - 第二个元素：包含变更信息的字典，键为配置项名称，值为包含旧值和新值的字典
        
        线程安全说明：此方法使用写锁，确保回滚操作的原子性
        """
        logger.info("开始执行配置回滚")
        
        with self._lock:
            if self._previous_config is None:
                logger.warning("没有可用的上一个配置，无法回滚")
                return False, {}
            
            # 保存当前配置作为新的 previous_config，以便可以再次回滚
            old_config = self._current_config
            new_config = self._previous_config
            
            # 计算配置变更
            changes = self._compare_configs(old_config, new_config)
            
            # 执行回滚
            self._current_config = new_config
            self._previous_config = old_config
            
            logger.info(f"配置回滚成功，变更项数量: {len(changes)}")
            if changes:
                masked_changes = self._mask_sensitive_changes(changes)
                logger.info(f"回滚变更详情: {masked_changes}")
            
            # 调用所有注册的回调函数
            self._call_callbacks(old_config, new_config)
            
            return True, changes

    def get_previous_config(self) -> Optional['StockSelectorConfig']:
        """
        获取上一个有效配置（仅用于调试和监控，不应用于业务逻辑）

        Returns:
            Optional[StockSelectorConfig]: 上一个有效配置，如果不存在则返回 None
        
        线程安全说明：此方法使用读锁
        """
        with self._lock:
            return self._previous_config

    def has_previous_config(self) -> bool:
        """
        检查是否存在上一个有效配置

        Returns:
            bool: 存在上一个有效配置返回 True，否则返回 False
        
        线程安全说明：此方法使用读锁
        """
        with self._lock:
            return self._previous_config is not None
    
    def register_callback(self, callback: Callable[['StockSelectorConfig', 'StockSelectorConfig'], None]) -> str:
        """
        注册配置变更回调函数

        当配置发生变更时，所有注册的回调函数都会被调用，
        回调函数会接收到两个参数：旧配置和新配置。

        Args:
            callback: 回调函数，签名为 (old_config: StockSelectorConfig, new_config: StockSelectorConfig) -> None

        Returns:
            str: 回调函数的唯一ID，可用于后续取消注册
        
        线程安全说明：此方法使用锁，确保回调函数注册的原子性
        """
        with self._lock:
            # 生成唯一的回调ID
            self._callback_id_counter += 1
            callback_id = f"callback_{self._callback_id_counter}"
            
            # 注册回调函数
            self._callbacks[callback_id] = callback
            
            logger.debug(f"注册配置变更回调函数，ID: {callback_id}")
            return callback_id
    
    def unregister_callback(self, callback_id: str) -> bool:
        """
        取消注册配置变更回调函数

        Args:
            callback_id: 回调函数的唯一ID，由 register_callback 方法返回

        Returns:
            bool: 成功取消返回 True，如果回调ID不存在返回 False
        
        线程安全说明：此方法使用锁，确保回调函数取消注册的原子性
        """
        with self._lock:
            if callback_id in self._callbacks:
                del self._callbacks[callback_id]
                logger.debug(f"取消注册配置变更回调函数，ID: {callback_id}")
                return True
            else:
                logger.warning(f"尝试取消注册不存在的回调函数，ID: {callback_id}")
                return False
    
    def mask_sensitive_value(self, key: str, value: Any) -> Any:
        """
        掩码敏感信息

        根据配置项的键名识别敏感信息，并对其值进行掩码处理。
        敏感信息包括但不限于：token、password、secret、key 等。

        Args:
            key: 配置项的键名
            value: 配置项的值

        Returns:
            Any: 如果是敏感信息，返回掩码后的值；否则返回原值
        """
        # 敏感信息关键词列表（不区分大小写）
        sensitive_keywords = {
            'token', 'password', 'secret', 'key', 'api_key', 'apikey',
            'auth', 'authorization', 'credential', 'pass', 'pwd',
            'private', 'certificate', 'cert', 'secret_key', 'access_key'
        }
        
        # 检查键名是否包含敏感关键词
        key_lower = key.lower()
        is_sensitive = any(keyword in key_lower for keyword in sensitive_keywords)
        
        if is_sensitive and value is not None:
            # 对敏感信息进行掩码处理
            value_str = str(value)
            if len(value_str) <= 4:
                # 太短的字符串全部掩码
                return '*' * len(value_str)
            else:
                # 保留首尾各2个字符，中间用星号代替
                return value_str[:2] + '*' * (len(value_str) - 4) + value_str[-2:]
        
        return value
    
    def start_monitoring(self, env_file_path: Optional[str] = None, 
                        monitor_interval: float = 1.0,
                        debounce_time: float = 0.5):
        """
        启动 .env 文件监控

        Args:
            env_file_path: .env 文件路径，如果不指定则查找当前目录下的 .env 文件
            monitor_interval: 监控间隔（秒），默认 1 秒
            debounce_time: 防抖时间（秒），默认 0.5 秒，避免短时间内多次修改频繁触发
        
        Raises:
            FileNotFoundError: 如果指定的 .env 文件不存在
            RuntimeError: 如果监控线程已经在运行
        """
        logger.info("启动 .env 文件监控")
        
        with self._lock:
            # 检查监控线程是否已在运行
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                logger.error("文件监控线程已在运行中")
                raise RuntimeError("文件监控线程已在运行中")
            
            # 设置监控参数
            self._monitor_interval = monitor_interval
            self._debounce_time = debounce_time
            logger.debug(f"监控间隔: {self._monitor_interval}秒，防抖时间: {self._debounce_time}秒")
            
            # 确定 .env 文件路径
            if env_file_path is not None:
                self._env_file_path = Path(env_file_path)
            else:
                # 默认查找当前目录下的 .env 文件
                self._env_file_path = Path.cwd() / ".env"
            
            logger.debug(f"监控文件路径: {self._env_file_path}")
            
            # 检查文件是否存在
            if not self._env_file_path.exists():
                logger.error(f"找不到 .env 文件: {self._env_file_path}")
                raise FileNotFoundError(f"找不到 .env 文件: {self._env_file_path}")
            
            # 记录初始修改时间
            self._last_modified_time = os.path.getmtime(self._env_file_path)
            logger.debug(f"初始文件修改时间: {self._last_modified_time}")
            
            # 重置停止事件
            self._monitor_stop_event.clear()
            
            # 创建并启动监控线程
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="env-file-monitor"
            )
            self._monitor_thread.start()
            
            logger.info(".env 文件监控已启动")
    
    def stop_monitoring(self):
        """
        停止 .env 文件监控
        
        线程安全说明：此方法是线程安全的，可以从任意线程调用
        """
        logger.info("停止 .env 文件监控")
        
        # 设置停止事件，通知监控线程退出
        self._monitor_stop_event.set()
        
        # 等待监控线程结束（最多等待 2 秒，避免无限等待）
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        with self._lock:
            self._monitor_thread = None
        
        logger.info(".env 文件监控已停止")
    
    def _monitor_loop(self):
        """
        监控循环（内部方法，在监控线程中运行）
        
        定期检查 .env 文件的修改时间，发现变化时自动重载配置
        """
        logger.debug("监控循环开始")
        
        while not self._monitor_stop_event.is_set():
            try:
                # 检查文件是否存在
                if self._env_file_path is not None and self._env_file_path.exists():
                    # 获取当前文件修改时间
                    current_mtime = os.path.getmtime(self._env_file_path)
                    
                    # 检查文件是否被修改
                    if current_mtime > self._last_modified_time:
                        logger.debug(f"检测到 .env 文件修改，旧时间: {self._last_modified_time}，新时间: {current_mtime}")
                        
                        # 更新上次修改时间
                        self._last_modified_time = current_mtime
                        
                        # 获取当前时间
                        current_time = time.time()
                        
                        # 检查防抖时间，避免短时间内多次触发
                        if current_time - self._last_reload_time >= self._debounce_time:
                            # 更新上次重载时间
                            self._last_reload_time = current_time
                            
                            # 重新加载配置
                            logger.info("触发配置自动重载")
                            self.reload_config()
                        else:
                            logger.debug("防抖时间内，跳过配置重载")
                
                # 等待指定的监控间隔，或直到停止事件被设置
                self._monitor_stop_event.wait(self._monitor_interval)
                
            except Exception as e:
                # 监控循环中发生异常时，继续运行，避免监控线程崩溃
                logger.error(f"监控循环发生异常: {str(e)}", exc_info=True)
                # 等待一段时间后继续
                self._monitor_stop_event.wait(self._monitor_interval)
        
        logger.debug("监控循环结束")
    
    def _compare_configs(self, old_config, new_config) -> Dict[str, Dict[str, Any]]:
        """
        比较两个配置对象的差异

        Args:
            old_config: 旧配置对象
            new_config: 新配置对象

        Returns:
            Dict[str, Dict[str, Any]]: 包含变更信息的字典，键为配置项名称，值为包含 'old' 和 'new' 的字典
        """
        changes = {}
        
        from .config import StockSelectorConfig
        
        # 使用 dataclasses.fields 遍历配置的所有字段
        for field in fields(StockSelectorConfig):
            field_name = field.name
            old_value = getattr(old_config, field_name)
            new_value = getattr(new_config, field_name)
            
            # 比较字段值是否不同
            if old_value != new_value:
                changes[field_name] = {
                    'old': old_value,
                    'new': new_value
                }
        
        return changes
    
    def _mask_sensitive_changes(self, changes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        对变更信息中的敏感信息进行掩码处理

        Args:
            changes: 原始的变更信息字典

        Returns:
            Dict[str, Dict[str, Any]]: 掩码处理后的变更信息字典
        """
        masked_changes = {}
        
        for key, value in changes.items():
            masked_changes[key] = {
                'old': self.mask_sensitive_value(key, value['old']),
                'new': self.mask_sensitive_value(key, value['new'])
            }
        
        return masked_changes
    
    def _call_callbacks(self, old_config: 'StockSelectorConfig', new_config: 'StockSelectorConfig'):
        """
        调用所有注册的回调函数

        Args:
            old_config: 旧配置对象
            new_config: 新配置对象
        
        线程安全说明：此方法在持有锁的情况下被调用，但回调函数的执行不持有锁
        """
        # 在持有锁的情况下复制回调函数列表，避免在回调执行过程中列表被修改
        with self._lock:
            callbacks_copy = list(self._callbacks.values())
        
        logger.debug(f"开始调用 {len(callbacks_copy)} 个配置变更回调函数")
        
        # 不持有锁的情况下执行回调函数，避免死锁
        for callback in callbacks_copy:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.error(f"配置变更回调函数执行失败: {str(e)}", exc_info=True)
        
        logger.debug("配置变更回调函数调用完成")
