# -*- coding: utf-8 -*-
"""
依赖注入容器测试脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.container import DependencyContainer, get_container, ServiceLifetime


class DatabaseService:
    """示例数据库服务"""
    def __init__(self):
        self.connected = False
    
    def connect(self):
        self.connected = True
        return "Database connected"
    
    def query(self, sql: str):
        return f"Query executed: {sql}"


class CacheService:
    """示例缓存服务"""
    def __init__(self):
        self.cache = {}
    
    def get(self, key: str):
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        self.cache[key] = value


class UserService:
    """示例用户服务"""
    def __init__(self):
        # 在这个简单示例中，我们手动依赖容器不做自动注入
        self.db_service: DatabaseService
        self.cache_service: CacheService
    
    def set_services(self, db: DatabaseService, cache: CacheService):
        self.db_service = db
        self.cache_service = cache
    
    def get_user(self, user_id: int):
        # 先查缓存
        cached = self.cache_service.get(f"user_{user_id}")
        if cached:
            return cached
        
        # 查数据库
        result = self.db_service.query(f"SELECT * FROM users WHERE id={user_id}")
        self.cache_service.set(f"user_{user_id}", result)
        return result


def test_basic_registration():
    """测试基本注册和解析"""
    print("=" * 80)
    print("测试 1: 基本注册和解析")
    print("=" * 80)
    
    container = DependencyContainer()
    
    # 注册单例服务
    container.register_singleton(DatabaseService)
    container.register_singleton(CacheService)
    
    # 解析服务
    db1 = container.resolve(DatabaseService)
    db2 = container.resolve(DatabaseService)
    
    assert db1 is db2, "单例应该返回同一实例"
    print(f"✅ 单例服务解析成功，是同一实例: {db1 is db2}")
    
    # 测试服务功能
    result = db1.connect()
    print(f"✅ 服务功能正常: {result}")
    
    print("\n✅ 基本注册和解析测试通过!")


def test_transient_registration():
    """测试瞬态服务"""
    print("\n" + "=" * 80)
    print("测试 2: 瞬态服务")
    print("=" * 80)
    
    container = DependencyContainer()
    
    # 注册瞬态服务
    container.register_transient(CacheService)
    
    # 解析服务
    cache1 = container.resolve(CacheService)
    cache2 = container.resolve(CacheService)
    
    assert cache1 is not cache2, "瞬态应该返回不同实例"
    print(f"✅ 瞬态服务解析成功，是不同实例: {cache1 is not cache2}")
    
    # 测试独立状态
    cache1.set("key1", "value1")
    cache2.set("key2", "value2")
    
    assert cache1.get("key1") == "value1"
    assert cache2.get("key1") is None
    print("✅ 瞬态服务状态独立")
    
    print("\n✅ 瞬态服务测试通过!")


def test_instance_registration():
    """测试实例注册"""
    print("\n" + "=" * 80)
    print("测试 3: 实例注册")
    print("=" * 80)
    
    container = DependencyContainer()
    
    # 创建实例
    db = DatabaseService()
    db.connect()
    
    # 注册实例
    container.register_instance(DatabaseService, db)
    
    # 解析实例
    resolved = container.resolve(DatabaseService)
    
    assert resolved is db, "应该返回注册的实例"
    assert resolved.connected, "实例状态应该保留"
    print(f"✅ 实例注册成功，返回同一实例: {resolved is db}")
    
    print("\n✅ 实例注册测试通过!")


def test_factory_registration():
    """测试工厂函数注册"""
    print("\n" + "=" * 80)
    print("测试 4: 工厂函数注册")
    print("=" * 80)
    
    container = DependencyContainer()
    
    # 创建一个计数器
    counter = 0
    
    def create_user_service(c):
        nonlocal counter
        counter += 1
        return UserService()
    
    # 注册工厂函数
    container.register_factory(UserService, create_user_service)
    
    # 解析服务
    user_service = container.resolve(UserService)
    
    assert isinstance(user_service, UserService)
    print("✅ 工厂函数注册成功，创建了服务实例")
    
    print("\n✅ 工厂函数注册测试通过!")


def test_get_with_default():
    """测试 get 方法"""
    print("\n" + "=" * 80)
    print("测试 5: get 方法")
    print("=" * 80)
    
    container = DependencyContainer()
    
    # 尝试获取未注册的服务
    result = container.get(DatabaseService, default=None)
    assert result is None, "未注册服务应该返回默认值"
    print("✅ get 方法正确处理未注册服务，返回默认值")
    
    # 注册后再获取
    container.register_singleton(DatabaseService)
    result = container.get(DatabaseService, default=None)
    assert result is not None, "已注册服务应该返回实例"
    print("✅ get 方法正确返回已注册服务")
    
    print("\n✅ get 方法测试通过!")


def test_global_container():
    """测试全局容器"""
    print("\n" + "=" * 80)
    print("测试 6: 全局容器")
    print("=" * 80)
    
    # 获取全局容器
    container1 = get_container()
    container2 = get_container()
    
    assert container1 is container2, "全局容器应该是同一实例"
    print("✅ 全局容器是同一实例")
    
    # 注册服务
    container1.register_singleton(DatabaseService)
    
    # 从另一个引用解析
    db = container2.resolve(DatabaseService)
    assert db is not None, "应该能解析到服务"
    print("✅ 全局容器服务解析成功")
    
    print("\n✅ 全局容器测试通过!")


def main():
    """主函数"""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "依赖注入容器测试" + " " * 40 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        test_basic_registration()
        test_transient_registration()
        test_instance_registration()
        test_factory_registration()
        test_get_with_default()
        test_global_container()
        
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 20 + "✅ 所有测试通过!" + " " * 40 + "║")
        print("╚" + "=" * 78 + "╝")
        
        print("\n📝 依赖注入容器功能:")
        print("   - 单例服务注册和解析")
        print("   - 瞬态服务注册和解析")
        print("   - 实例直接注册")
        print("   - 工厂函数注册")
        print("   - get 方法（带默认值）")
        print("   - 全局容器单例")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
