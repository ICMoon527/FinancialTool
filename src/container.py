# -*- coding: utf-8 -*-
"""
===================================
依赖注入容器
===================================

职责：
1. 管理服务实例的生命周期
2. 提供服务解析
3. 支持单例和瞬态服务
"""

import logging
from enum import Enum
from typing import Dict, Type, Any, Callable, Optional, TypeVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """服务生命周期"""
    SINGLETON = "singleton"  # 单例
    TRANSIENT = "transient"  # 瞬态（每次创建新实例）
    SCOPED = "scoped"  # 作用域（每个请求一个实例）


@dataclass
class ServiceDescriptor:
    """服务描述符"""
    service_type: Type
    implementation_type: Optional[Type] = None
    instance: Optional[Any] = None
    factory: Optional[Callable] = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON


class DependencyContainer:
    """
    依赖注入容器
    
    简单的 DI 容器实现，支持：
    - 单例服务注册
    - 瞬态服务注册
    - 工厂函数注册
    - 实例注册
    """
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_services: Dict[str, Dict[Type, Any]] = {}
    
    def register_singleton(self, service_type: Type, implementation: Optional[Type] = None) -> 'DependencyContainer':
        """
        注册单例服务
        
        Args:
            service_type: 服务类型
            implementation: 实现类型（如果为 None，则使用 service_type）
        
        Returns:
            DependencyContainer: 自身（支持链式调用）
        """
        impl_type = implementation or service_type
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=impl_type,
            lifetime=ServiceLifetime.SINGLETON
        )
        logger.debug(f"Registered singleton: {service_type.__name__} -> {impl_type.__name__}")
        return self
    
    def register_transient(self, service_type: Type, implementation: Optional[Type] = None) -> 'DependencyContainer':
        """
        注册瞬态服务
        
        Args:
            service_type: 服务类型
            implementation: 实现类型
        
        Returns:
            DependencyContainer: 自身
        """
        impl_type = implementation or service_type
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=impl_type,
            lifetime=ServiceLifetime.TRANSIENT
        )
        logger.debug(f"Registered transient: {service_type.__name__} -> {impl_type.__name__}")
        return self
    
    def register_factory(self, service_type: Type, factory: Callable[['DependencyContainer'], Any]) -> 'DependencyContainer':
        """
        注册工厂函数
        
        Args:
            service_type: 服务类型
            factory: 工厂函数，接受容器参数，返回服务实例
        
        Returns:
            DependencyContainer: 自身
        """
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            factory=factory,
            lifetime=ServiceLifetime.TRANSIENT
        )
        logger.debug(f"Registered factory: {service_type.__name__}")
        return self
    
    def register_instance(self, service_type: Type, instance: Any) -> 'DependencyContainer':
        """
        注册现有实例
        
        Args:
            service_type: 服务类型
            instance: 服务实例
        
        Returns:
            DependencyContainer: 自身
        """
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
        self._singletons[service_type] = instance
        logger.debug(f"Registered instance: {service_type.__name__}")
        return self
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        解析服务
        
        Args:
            service_type: 服务类型
        
        Returns:
            服务实例
        
        Raises:
            KeyError: 如果服务未注册
        """
        if service_type not in self._services:
            raise KeyError(f"Service not registered: {service_type.__name__}")
        
        descriptor = self._services[service_type]
        
        # 如果已有实例，直接返回
        if descriptor.instance is not None:
            return descriptor.instance
        
        # 单例模式
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]
            
            instance = self._create_instance(descriptor)
            self._singletons[service_type] = instance
            return instance
        
        # 瞬态模式
        return self._create_instance(descriptor)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        # 使用工厂函数
        if descriptor.factory is not None:
            return descriptor.factory(self)
        
        # 使用实现类型
        if descriptor.implementation_type is not None:
            # 尝试自动注入构造函数参数
            impl_type = descriptor.implementation_type
            
            # 简单的构造函数注入（仅支持无参构造函数）
            # 更复杂的 DI 需要使用专门的库
            return impl_type()
        
        raise ValueError(f"Cannot create instance for {descriptor.service_type.__name__}")
    
    def get(self, service_type: Type[T], default: Optional[T] = None) -> Optional[T]:
        """
        尝试解析服务，不存在时返回默认值
        
        Args:
            service_type: 服务类型
            default: 默认值
        
        Returns:
            服务实例或默认值
        """
        try:
            return self.resolve(service_type)
        except KeyError:
            return default


# 全局容器实例
_container: Optional[DependencyContainer] = None


def get_container() -> DependencyContainer:
    """获取全局容器实例"""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container


def set_container(container: DependencyContainer) -> None:
    """设置全局容器实例"""
    global _container
    _container = container
