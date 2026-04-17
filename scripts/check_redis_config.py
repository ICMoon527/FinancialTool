# -*- coding: utf-8 -*-
"""
检测 Redis 配置是否正确加载
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.cache import CacheManager


def check_redis_config():
    """检查 Redis 配置"""
    print("=" * 80)
    print("Redis 配置检测")
    print("=" * 80)
    
    config = get_config()
    
    print("\n1. 配置文件加载:")
    print(f"   REDIS_ENABLED: {config.redis_enabled}")
    print(f"   REDIS_HOST: {config.redis_host}")
    print(f"   REDIS_PORT: {config.redis_port}")
    print(f"   REDIS_DB: {config.redis_db}")
    print(f"   REDIS_PASSWORD: {'已设置' if config.redis_password else '未设置'}")
    print(f"   REDIS_TTL_SECONDS: {config.redis_ttl_seconds}")
    print(f"   REDIS_MAX_CACHE_ENTRIES: {config.redis_max_cache_entries}")
    
    if not config.redis_enabled:
        print("\n⚠️  Redis 未启用！")
        print("   请在 .env 文件中设置 REDIS_ENABLED=true")
        return False
    
    print("\n✅ Redis 已启用！")
    
    print("\n2. 测试缓存管理器:")
    try:
        cache = CacheManager()
        
        if cache._redis_client:
            print("   ✅ Redis 连接成功！")
            
            # 测试简单操作
            cache.set("test:redis:config", "success")
            result = cache.get("test:redis:config")
            
            if result == "success":
                print("   ✅ Redis 读写测试通过！")
            else:
                print("   ❌ Redis 读写测试失败！")
            
            # 获取 Redis 信息
            stats = cache.get_stats()
            print(f"\n3. Redis 统计信息:")
            if 'redis_info' in stats:
                info = stats['redis_info']
                print(f"   Redis 版本: {info.get('redis_version', 'N/A')}")
                print(f"   已连接客户端: {info.get('connected_clients', 'N/A')}")
                print(f"   使用内存: {info.get('used_memory_human', 'N/A')}")
                print(f"   键总数: {info.get('db0', {}).get('keys', 'N/A')}")
            
            # 清理测试数据
            cache.delete("test:redis:config")
            
        else:
            print("   ⚠️  Redis 连接失败，使用 LRU 内存缓存")
            print("   请检查:")
            print("   1. Redis 是否已安装并启动？")
            print("   2. 端口配置是否正确？（默认 6379）")
            print("   3. 密码配置是否正确？")
    
    except Exception as e:
        print(f"   ❌ 缓存管理器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("检测完成！")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    check_redis_config()
