# 股票推荐系统 (FinancialTool)

A股量化交易回测与智能推荐平台，整合数据管理、策略研究、回测引擎、股票推荐于一体。

## 功能特性

### 1. 数据管理模块
- **多数据源集成**: 支持 AKShare、Tushare 双数据源，优雅降级机制
- **全市场覆盖**: 上证主板、深证主板、创业板、科创板、北交所所有股票
- **历史数据**: 日线数据获取
- **数据清洗**: 自动处理缺失值、异常值
- **数据存储**: SQLite 高效存储
- **智能缓存**: 自动缓存数据，避免重复获取，交易日结束后自动获取最新收盘价

### 2. 股票池管理模块
- **全板块支持**: 上证、深证、创业板、科创板、北交所
- **风险等级配置**: 四种风险等级适配不同投资者
  - 保守型：上证主板 + 深证主板
  - 稳健型：上证主板 + 深证主板 + 创业板
  - 进取型：上证主板 + 深证主板 + 创业板 + 科创板
  - 激进型：全部板块（含北交所）
- **动态筛选**: 根据风险等级自动筛选对应股票池

### 3. 策略研究模块
- **因子库**: 10+种常用技术指标（MA、EMA、MACD、RSI、布林带、KDJ、ATR、OBV等）
- **信号生成**: 7种内置策略（均线交叉、MACD、RSI、布林带、KDJ、Dual Thrust、海龟交易）
- **交互式开发**: 支持 Jupyter Notebook 快速迭代

### 4. 智能推荐引擎
- **多因子综合评分**: 技术面、动量、趋势、波动率多维度评分
- **三种持仓周期**: 短线、中线、长线推荐
- **智能筛选**: 自动排除ST、停牌、次新股
- **推荐理由**: 详细说明推荐原因和买入分析

### 5. 回测引擎模块
- **事件驱动**: 基于历史数据模拟市场事件
- **交易仿真**: 支持滑点、手续费、持仓变动精确模拟
- **绩效分析**: 夏普比率、最大回撤、年化收益等指标
- **参数优化**: 网格搜索、滚动窗口分析

## 项目结构

```
FinancialTool/
├── data/                  # 数据管理模块
│   ├── __init__.py
│   ├── database.py       # 数据库模型和管理
│   ├── data_fetcher.py   # 数据获取（含智能缓存）
│   ├── data_cleaner.py   # 数据清洗
│   ├── stock_universe.py # 股票池管理（含风险等级配置）
│   └── sample_data.py    # 示例数据
├── strategy/              # 策略研究模块
│   ├── __init__.py
│   ├── factors.py        # 因子库
│   ├── signals.py        # 信号生成
│   └── recommendation_engine.py # 智能推荐引擎
├── backtest/              # 回测引擎模块
│   ├── __init__.py
│   ├── engine.py         # 回测引擎
│   └── analytics.py      # 绩效分析和参数优化
├── examples/              # 示例代码
│   └── example_strategy.py
├── notebooks/             # Jupyter Notebooks
├── logs/                  # 日志目录（自动创建）
├── config.py              # 配置文件
├── logger.py              # 日志模块
├── requirements.txt       # 依赖包
├── .env.example           # 环境变量示例
└── main.py                # 主程序入口（交互式菜单）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写配置：

```bash
cp .env.example .env
```

**重要**: 系统优先使用 AKShare（无需Token），如果需要使用 Tushare 的完整功能，请在 https://tushare.pro 注册并获取 Token，填写到 `.env` 文件中。

### 3. 运行主程序

```bash
python main.py
```

运行后进入交互式菜单，可选择：
- 查看/修改股票池（选择风险等级）
- 获取短线推荐（明日上涨概率最高的5支股票）
- 获取中线推荐（中长期持有的5支股票）
- 获取长线推荐（长期持有的5支股票）

### 4. 运行策略回测示例

```bash
python -m examples.example_strategy
```

### 5. 启动 Jupyter Notebook

```bash
jupyter notebook
```

## 使用示例

### 股票池管理

```python
from data.stock_universe import stock_universe

# 获取风险等级信息
risk_info = stock_universe.get_risk_level_info()
print(risk_info)

# 根据风险等级获取股票池
# 1-保守型, 2-稳健型, 3-进取型, 4-激进型
stock_pool = stock_universe.get_stock_pool(risk_level=2)
print(f"股票池数量: {len(stock_pool)}")
```

### 股票推荐

```python
from strategy.recommendation_engine import recommendation_engine
from data.stock_universe import stock_universe

# 获取股票池
stock_pool = stock_universe.get_stock_pool(risk_level=2)

# 获取短线推荐（明日上涨概率最高的5支）
short_term = recommendation_engine.recommend_stocks(
    stock_pool, 
    horizon='short', 
    top_n=5
)

# 打印推荐结果
for rec in short_term:
    print(f"股票: {rec['name']}({rec['ts_code']})")
    print(f"综合评分: {rec['score']:.2f}")
    print(f"推荐理由: {rec['reason']}")
    print("-" * 50)
```

### 数据获取（智能缓存）

```python
from data.data_fetcher import data_fetcher

# 获取股票列表
stock_list = data_fetcher.fetch_stock_list()
data_fetcher.save_stock_list(stock_list)

# 获取单只股票日线数据（自动使用缓存）
df = data_fetcher.fetch_stock_daily('000001.SZ', '20200101', '20250101', use_cache=True)
data_fetcher.save_stock_daily('000001.SZ', df)
```

### 策略回测

```python
from strategy.factors import factor_library
from strategy.signals import signal_generator
from backtest.engine import backtest_engine
from backtest.analytics import performance_analyzer

# 计算技术指标
df = factor_library.calculate_all_factors(df)

# 生成交易信号
df = signal_generator.generate_signals(df, strategy_name='ma_cross')

# 运行回测
backtest_engine.reset()
results = backtest_engine.run(df, ts_code='000001.SZ')

# 打印绩效报告
print(performance_analyzer.generate_report(results))
```

### 参数优化

```python
from backtest.analytics import parameter_optimizer

param_grid = {
    'short_period': [3, 5, 8, 10],
    'long_period': [15, 20, 30, 60]
}

opt_result = parameter_optimizer.grid_search(
    strategy_func,
    param_grid,
    df,
    objective='sharpe_ratio'
)

print(f"最优参数: {opt_result['best_params']}")
```

## 内置策略

| 策略名称 | 描述 |
|---------|------|
| ma_cross | 均线交叉策略（金叉买，死叉卖） |
| macd | MACD 策略 |
| rsi | RSI 超买超卖策略 |
| bollinger | 布林带突破策略 |
| kdj | KDJ 策略 |
| dual_thrust | Dual Thrust 通道突破策略 |
| turtle | 海龟交易策略 |

## 注意事项

1. **数据质量**: 请确保数据完整性，系统已内置智能缓存机制
2. **缓存策略**: 交易日结束后（15:00后）自动获取当天最新收盘价
3. **未来函数**: 策略开发时请注意避免未来函数
4. **过拟合风险**: 参数优化时请使用样本外数据验证
5. **实盘风险**: 推荐结果和回测结果不代表实盘表现，请谨慎使用
6. **风险等级**: 根据自身风险承受能力选择合适的股票池

## 技术栈

- **数据获取**: AKShare（优先）, Tushare（备用）
- **数据处理**: Pandas, NumPy
- **数据库**: SQLAlchemy + SQLite
- **可视化**: Matplotlib, Plotly
- **交互式开发**: Jupyter Notebook
- **智能推荐**: 多因子综合评分系统

## 许可证

MIT License
