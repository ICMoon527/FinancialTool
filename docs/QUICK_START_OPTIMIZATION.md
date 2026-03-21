# 性能优化快速使用指南

## 🎉 优化已完成！

所有性能优化已经成功实现并通过测试。

---

## 📋 已添加的优化方法

### 1. `get_data_range_optimized()` - 带 LRU 缓存的数据查询

**位置**: [src/storage.py#L1090-L1140](file://e:\工作\Code\FinancialTool\src\storage.py#L1090-L1140)

**功能**:
- ✅ LRU 缓存机制（最近最少使用）
- ✅ 避免重复查询相同数据
- ✅ 5 分钟缓存 TTL（可配置）
- ✅ 最多缓存 1000 条记录（可配置）

**使用示例**:
```python
from src.storage import DatabaseManager

db = DatabaseManager.get_instance()

# 使用缓存查询
data = db.get_data_range_optimized('600519', start_date, end_date, use_cache=True)

# 不使用缓存查询
data = db.get_data_range_optimized('600519', start_date, end_date, use_cache=False)
```

**预期性能提升**: 
- 重复查询：**1000 倍+**（数据库访问 → 内存访问）
- 批量 screen：**70-80%**

---

### 2. `save_daily_data_bulk()` - 批量保存（解决 N+1 问题）

**位置**: [src/storage.py#L1186-L1300](file://e:\工作\Code\FinancialTool\src\storage.py#L1186-L1300)

**功能**:
- ✅ 批量查询已存在的记录（1 次查询代替 N 次）
- ✅ 批量插入新记录
- ✅ 批量更新现有记录
- ✅ 自动清除相关缓存

**使用示例**:
```python
from src.storage import DatabaseManager
import pandas as pd

db = DatabaseManager.get_instance()

# 批量保存数据
count = db.save_daily_data_bulk(df, '600519', data_source='tushare')
```

**预期性能提升**:
- 100 条数据：100 次查询 → 1 次查询
- 性能提升：**99%**

---

### 3. `get_data_range_with_fields()` - 字段投影（减少数据传输）

**位置**: [src/storage.py#L1142-L1184](file://e:\工作\Code\FinancialTool\src\storage.py#L1142-L1184)

**功能**:
- ✅ 只检索需要的字段
- ✅ 直接返回 DataFrame
- ✅ 默认使用核心 7 个字段

**使用示例**:
```python
from src.storage import DatabaseManager

db = DatabaseManager.get_instance()

# 使用默认字段（date, open, high, low, close, volume, pct_chg）
df = db.get_data_range_with_fields('600519', start_date, end_date)

# 使用自定义字段
df = db.get_data_range_with_fields('600519', start_date, end_date, 
                                   fields=['date', 'close', 'volume'])
```

**预期性能提升**:
- 数据传输量减少：**~53%**
- 内存占用减少：**~30%**

---

### 4. 缓存管理方法

#### `_invalidate_cache()` - 清除指定股票缓存

**使用示例**:
```python
db._invalidate_cache('600519')
```

#### `clear_cache()` - 清除所有缓存

**使用示例**:
```python
db.clear_cache()
```

---

## 🚀 在现有代码中使用优化方法

### 场景 1: Screen 选股（使用缓存）

**优化前**:
```python
# stock_selector/strategies/python/*.py
data = db.get_data_range(stock_code, start_date, end_date)
```

**优化后**:
```python
# stock_selector/strategies/python/*.py
data = db.get_data_range_optimized(stock_code, start_date, end_date, use_cache=True)
```

---

### 场景 2: 批量更新数据（解决 N+1）

**优化前**:
```python
# stock_selector/data_validator.py, stock_selector/batch_data_updater.py
db.save_daily_data(df, stock_code)
```

**优化后**:
```python
# stock_selector/data_validator.py, stock_selector/batch_data_updater.py
db.save_daily_data_bulk(df, stock_code, data_source)
```

---

### 场景 3: 只需要部分字段（减少传输）

**优化前**:
```python
data = db.get_data_range(stock_code, start_date, end_date)
# 获取所有 15 个字段
```

**优化后**:
```python
df = db.get_data_range_with_fields(stock_code, start_date, end_date, 
                                   fields=['date', 'close', 'volume'])
# 只获取 3 个字段
```

---

## 📊 性能对比总结

| 操作 | 优化方法 | 预期提升 |
|------|----------|----------|
| **数据查询（重复）** | `get_data_range_optimized()` | **1000 倍+** |
| **批量 screen** | `get_data_range_optimized()` | **70-80%** |
| **数据保存** | `save_daily_data_bulk()` | **99%** |
| **字段传输** | `get_data_range_with_fields()` | **~53%** |
| **内存占用** | `get_data_range_with_fields()` | **~30%** |

---

## 🔧 测试文件

1. **[tests/performance_benchmark.py](file://e:\工作\Code\FinancialTool\tests\performance_benchmark.py)**
   - 完整的性能基准测试
   - 可用于优化前后对比

2. **[tests/validate_optimizations.py](file://e:\工作\Code\FinancialTool\tests\validate_optimizations.py)**
   - 快速验证测试（已通过 ✅）
   - 用于检查优化方法是否正常工作

---

## 📝 文档

- **[docs/PERFORMANCE_OPTIMIZATION_GUIDE.md](file://e:\工作\Code\FinancialTool\docs\PERFORMANCE_OPTIMIZATION_GUIDE.md)**
  - 详细的性能优化指南
  - 包含问题分析、优化方案、使用示例

---

## ⚠️ 注意事项

### 1. 缓存一致性
- 保存数据后会自动清除相关缓存
- 如果手动更新数据库，记得调用 `db.clear_cache()`

### 2. 内存管理
- 默认缓存 1000 条记录，5 分钟 TTL
- 可根据需要调整：
  ```python
  db.CACHE_MAX_SIZE = 500    # 减少缓存条目
  db.CACHE_TTL_SECONDS = 120   # 缩短缓存时间
  ```

### 3. 向后兼容
- 所有原方法保持不变
- 可以渐进式替换使用优化方法

---

## 🎯 下一步

1. 在关键路径使用优化方法（如 screen 选股、批量更新数据）
2. 运行性能基准测试进行对比
3. 监控性能提升效果
4. 根据实际使用情况调整缓存参数
