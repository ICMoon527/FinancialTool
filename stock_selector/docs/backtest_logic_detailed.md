# 六维选股策略回测系统 - 详细逻辑文档

## 1. 系统概述

### 1.1 系统目标
本回测系统旨在对六维选股策略及其他投资策略进行历史回测，验证策略的有效性、稳定性和风险收益特征，为策略优化和实盘交易提供科学依据。

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        回测系统架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐  │
│  │ 配置管理模块  │────▶│ 回测引擎模块  │◀────│ 数据提供模块  │  │
│  └──────────────┘     └──────┬───────┘     └──────────────┘  │
│                               │                                   │
│  ┌──────────────┐     ┌──────▼───────┐     ┌──────────────┐  │
│  │ 策略管理模块  │────▶│ 回测编排器   │◀────│ 股票池管理   │  │
│  └──────────────┘     └──────┬───────┘     └──────────────┘  │
│                               │                                   │
│  ┌──────────────┐     ┌──────▼───────┐     ┌──────────────┐  │
│  │ 结果可视化   │◀────│ 结果评估模块  │◀────│ 指标计算模块 │  │
│  └──────────────┘     └──────────────┘     └──────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 核心模块详解

### 2.1 回测核心引擎 (BacktestEngine)

**文件位置**: `src/core/backtest_engine.py`

#### 2.1.1 核心功能

回测引擎是整个系统的核心，负责单个策略分析的历史回测评估。

#### 2.1.2 主要类和方法

##### 2.1.2.1 EvaluationConfig
```python
@dataclass(frozen=True)
class EvaluationConfig:
    eval_window_days: int          # 评估窗口天数
    neutral_band_pct: float = 2.0  # 中性区间百分比
    engine_version: str = "v1"     # 引擎版本号
```

##### 2.1.2.2 BacktestEngine 核心方法

**infer_direction_expected()**: 推断预期方向

根据操作建议推断股票预期走势方向：
- "买入"、"加仓"等 → "up"（上涨）
- "卖出"、"减仓"等 → "down"（下跌）
- "持有" → "not_down"（不跌）
- "观望"、"等待" → "flat"（持平）

**infer_position_recommendation()**: 推断持仓建议

根据操作建议推断推荐持仓：
- "买入"、"加仓"、"持有" → "long"（持有）
- "卖出"、"观望" → "cash"（现金）
- 无法识别 → "cash"（保守策略）

**evaluate_single()**: 单个评估执行

这是回测引擎的核心方法，包含完整的回测逻辑：

```
输入参数:
- operation_advice: 操作建议
- analysis_date: 分析日期
- start_price: 起始价格
- forward_bars: 未来K线数据序列
- stop_loss: 止损价格
- take_profit: 止盈价格
- config: 评估配置

执行步骤:
1. 数据有效性验证
   - 检查start_price是否有效
   - 检查评估窗口是否为正数
   - 检查是否有足够的未来数据

2. 入场价格确定
   - 使用T+1规则：不能在买入当天卖出
   - 如果有Day 1开盘价，使用Day 1开盘价作为入场价
   - 否则使用start_price作为入场价

3. 基础收益计算
   - 计算股票在评估窗口内的实际收益率
   - stock_return_pct = (end_close - entry_price) / entry_price * 100

4. 方向和结果分类
   - 根据预期方向和实际收益分类为win/loss/neutral
   - 判断方向是否正确

5. 止损止盈评估
   - 从Day 2开始检查止损止盈
   - 如果Day 2开盘价直接触发止损止盈，使用该价格
   - 记录首次触发的时间和价格
   - 处理同一天同时触发止损止盈的歧义情况

6. 模拟交易收益计算
   - 如果是long持仓，计算模拟交易收益
   - 如果是cash持仓，收益为0

7. 返回完整的评估结果
```

**compute_summary()**: 结果汇总

将多个回测结果聚合为统计指标：
- 总评估数、完成数、数据不足数
- 持仓分布（long/cash）
- 胜负平分布
- 胜率、方向准确率
- 平均股票收益、平均模拟交易收益
- 止损触发率、止盈触发率
- 首次触发平均天数
- 操作建议分类统计
- 诊断信息

### 2.2 股票池管理模块

**文件位置**: `stock_selector/stock_pool.py`

#### 2.2.1 核心功能

管理回测使用的股票池，确保与stock_selector保持一致的范围。

#### 2.2.2 股票池范围

**沪A主板 + 深A主板**

包含：
- 60开头：沪市主板
- 00开头：深市主板

过滤规则：
- ❌ 688开头：科创板
- ❌ 300/301开头：创业板
- ❌ 43/83/87/92开头：北交所
- ❌ ST/*ST/SST/S*ST股票

#### 2.2.3 主要函数

**get_all_stock_code_name_pairs()**: 获取所有股票代码和名称

从数据源获取完整股票列表，并缓存到数据库。

**filter_special_stock_codes()**: 过滤特定板块股票

过滤科创板、创业板、北交所股票。

**filter_st_stocks()**: 过滤ST股票

根据股票名称过滤ST股票。

**filter_beijing_stock_exchange()**: 过滤北交所股票

专门过滤8开头和92开头的北交所股票。

### 2.3 回测编排器 (BacktestOrchestrator)

**文件位置**: 需进一步确认（通常在src/core/strategy_backtest.py）

#### 2.3.1 核心功能

负责编排完整的回测流程，从数据获取、策略执行到结果保存。

#### 2.3.2 主要方法

**run_full_backtest()**: 执行完整回测

```
执行步骤:
1. 加载配置文件
2. 初始化数据提供器
3. 初始化策略
4. 获取或构建股票池
5. 遍历回测时间范围内的每个交易日
6. 对每只股票执行策略分析
7. 使用回测引擎评估策略表现
8. 收集所有回测结果
9. 计算汇总指标
10. 生成可视化图表
11. 保存回测报告
```

### 2.4 配置管理模块

**文件位置**: `stock_selector/backtest_config.yaml`

#### 2.4.1 配置参数详解

```yaml
# 初始资金（元）
initial_capital: 1000000.0

# 手续费率（万分之三）
commission_rate: 0.0003

# 滑点率（千分之一）
slippage_rate: 0.001

# 无风险利率（年化3%）
# 无风险利率是指将资金投资于某一项没有任何风险的投资对象而能得到的利息率。
# 在回测中，无风险利率用于计算夏普比率、卡尔马比率等风险调整收益指标。
# 通常使用一年期国债收益率或银行定期存款利率作为无风险利率。
risk_free_rate: 0.03

# 回测开始日期（YYYY-MM-DD格式）
start_date: "2025-01-01"

# 回测结束日期（YYYY-MM-DD格式）
end_date: "2025-12-31"

# 股票池范围：沪A主板 + 深A主板
# 过滤规则（与stock_selector一致）：
# - 保留：60开头（沪市主板）、00开头（深市主板）
# - 过滤：688开头（科创板）、300/301开头（创业板）、8/43/83/87/92开头（北交所）、ST股票
# 注：如果设置为空列表[]，回测系统将自动从stock_pool获取完整的沪A+深A主板股票列表
stock_pool: []
```

#### 2.4.2 无风险利率说明

**什么是无风险利率？**

无风险利率是指将资金投资于某一项没有任何风险的投资对象而能得到的利息率。它是投资收益的基准利率，代表了投资者在不承担任何风险的情况下所能获得的最低回报率。

**在回测中的作用：**

1. **计算夏普比率 (Sharpe Ratio)**:
   ```
   夏普比率 = (策略年化收益率 - 无风险利率) / 策略年化波动率
   ```
   夏普比率衡量每承担一单位风险所能获得的超额收益。

2. **计算卡尔马比率 (Calmar Ratio)**:
   ```
   卡尔马比率 = 策略年化收益率 / 最大回撤
   ```
   有时也会使用超额收益来计算。

3. **计算信息比率 (Information Ratio)**:
   比较策略收益相对于基准的超额收益。

4. **机会成本参考**:
   无风险利率代表了资金的机会成本，帮助判断策略是否值得投资。

**常见的无风险利率选择：**
- 中国：一年期国债收益率（约2%-3%）、一年期银行定期存款利率
- 美国：3个月期国债收益率（T-bill）

## 3. 回测执行流程

### 3.1 完整执行流程图

```
开始
  │
  ▼
┌─────────────────────────────┐
│ 1. 加载配置文件             │
│    - 读取backtest_config.yaml│
│    - 解析回测参数           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 2. 构建股票池               │
│    - 如果stock_pool为空     │
│      - 从stock_pool获取     │
│      - 过滤特定板块         │
│      - 过滤ST股票           │
│    - 否则使用配置的股票池   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 3. 初始化组件               │
│    - 数据提供器             │
│    - 策略实例               │
│    - 回测引擎               │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 4. 遍历回测日期范围         │
│    对每个交易日:            │
│    ┌─────────────────────┐  │
│    │ 4.1 获取当前股票池  │  │
│    └──────────┬──────────┘  │
│               │              │
│    ┌──────────▼──────────┐  │
│    │ 4.2 对每只股票执行  │  │
│    │     策略分析        │  │
│    └──────────┬──────────┘  │
│               │              │
│    ┌──────────▼──────────┐  │
│    │ 4.3 使用回测引擎    │  │
│    │     评估策略表现    │  │
│    └──────────┬──────────┘  │
│               │              │
│    └──────────▼──────────┘  │
│      4.4 保存评估结果      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 5. 计算汇总指标             │
│    - 胜率、收益率           │
│    - 最大回撤、夏普比率     │
│    - 风险调整收益指标       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 6. 生成可视化图表           │
│    - 资金曲线               │
│    - 回撤曲线               │
│    - 月度收益分布           │
│    - 胜率热力图             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 7. 保存回测报告             │
│    - JSON格式原始数据       │
│    - HTML格式可视化报告     │
│    - Markdown格式总结       │
└──────────────┬──────────────┘
               │
               ▼
              结束
```

### 3.2 单只股票回测详细逻辑

对于每只股票的单次策略分析，回测引擎执行以下详细步骤：

#### 步骤1：数据验证

```python
# 检查起始价格是否有效
if start_price is None or start_price <= 0:
    return error状态

# 检查评估窗口是否有效
if eval_window_days <= 0:
    raise ValueError

# 检查是否有足够的未来数据
if len(forward_bars) < eval_window_days:
    return insufficient_data状态
```

#### 步骤2：确定入场价格（T+1规则）

```python
# T+1交易规则：不能在买入当天卖出
# 使用Day 1的开盘价作为入场价格（如果可用）
entry_price = start_price
if window_bars and hasattr(window_bars[0], 'open') and window_bars[0].open is not None:
    entry_price = window_bars[0].open
```

#### 步骤3：计算股票基础收益

```python
# 获取评估窗口结束时的收盘价
end_close = window_bars[-1].close

# 计算股票在评估窗口内的实际收益率
if end_close is not None:
    stock_return_pct = (end_close - entry_price) / entry_price * 100
```

#### 步骤4：推断方向和持仓

```python
# 根据操作建议推断预期方向
direction_expected = infer_direction_expected(operation_advice)

# 根据操作建议推断持仓建议
position = infer_position_recommendation(operation_advice)
```

#### 步骤5：分类结果和判断方向正确性

```python
# 根据预期方向和实际收益进行分类
if direction_expected == "up":
    if stock_return_pct >= neutral_band_pct:
        outcome = "win"
        direction_correct = True
    elif stock_return_pct <= -neutral_band_pct:
        outcome = "loss"
        direction_correct = False
    else:
        outcome = "neutral"
        direction_correct = None
```

#### 步骤6：评估止损止盈（从Day 2开始）

```python
# 遍历评估窗口中的每一天（从Day 2开始）
for idx, bar in enumerate(window_bars, start=1):
    if idx == 1:
        continue  # 跳过Day 1
    
    low = bar.low
    high = bar.high
    open_price = bar.open
    
    # 检查开盘价是否直接触发止损止盈
    if open_price is not None:
        if stop_loss is not None and open_price <= stop_loss:
            stop_hit = True
            exit_price = open_price
        if take_profit is not None and open_price >= take_profit:
            tp_hit = True
            exit_price = open_price
    
    # 检查当日最低价和最高价
    if not stop_hit and not tp_hit:
        stop_hit = stop_loss is not None and low <= stop_loss
        tp_hit = take_profit is not None and high >= take_profit
        if stop_hit:
            exit_price = stop_loss
        if tp_hit:
            exit_price = take_profit
    
    # 如果触发了止损或止盈，记录并退出
    if stop_hit or tp_hit:
        first_hit_date = bar.date
        first_hit_days = idx
        
        # 处理同一天同时触发的歧义情况
        if stop_hit and tp_hit:
            first_hit = "ambiguous"
            if open_price >= take_profit:
                exit_reason = "ambiguous_take_profit"
            else:
                exit_reason = "ambiguous_stop_loss"
        elif stop_hit:
            first_hit = "stop_loss"
            exit_reason = "stop_loss"
        else:
            first_hit = "take_profit"
            exit_reason = "take_profit"
        break
```

#### 步骤7：计算模拟交易收益

```python
if position == "long":
    if simulated_exit_price is not None:
        simulated_return_pct = (simulated_exit_price - entry_price) / entry_price * 100
else:
    # 现金持仓，收益为0
    simulated_return_pct = 0.0
```

## 4. 评估指标详解

### 4.1 基础指标

| 指标名称 | 计算公式 | 说明 |
|---------|---------|------|
| 总评估数 | total_evaluations | 所有回测评估的总数 |
| 完成数 | completed_count | 成功完成的评估数 |
| 数据不足数 | insufficient_count | 因数据不足失败的评估数 |
| Long持仓数 | long_count | 推荐持有股票的评估数 |
| Cash持仓数 | cash_count | 推荐持有现金的评估数 |

### 4.2 收益指标

| 指标名称 | 计算公式 | 说明 |
|---------|---------|------|
| 胜率 | win_count / (win_count + loss_count) × 100% | 盈利交易占比 |
| 平均股票收益 | avg(stock_return_pct) | 所有完成评估的股票平均收益率 |
| 平均模拟交易收益 | avg(simulated_return_pct) | 模拟交易的平均收益率 |
| 方向准确率 | direction_numerator / direction_denominator × 100% | 方向判断正确的比例 |

### 4.3 风控指标

| 指标名称 | 计算公式 | 说明 |
|---------|---------|------|
| 止损触发率 | stop_loss_trigger_count / stop_applicable_count × 100% | 止损被触发的比例 |
| 止盈触发率 | take_profit_trigger_count / take_profit_applicable_count × 100% | 止盈被触发的比例 |
| 平均首次触发天数 | avg(first_hit_trading_days) | 首次触发止损止盈的平均天数 |
| 歧义率 | ambiguous_count / any_target_applicable_count × 100% | 同一天同时触发止损止盈的比例 |

### 4.4 风险调整收益指标（如适用）

| 指标名称 | 计算公式 | 说明 |
|---------|---------|------|
| 夏普比率 | (年化收益率 - 无风险利率) / 年化波动率 | 单位风险的超额收益 |
| 卡尔马比率 | 年化收益率 / 最大回撤 | 单位回撤的收益 |
| 索提诺比率 | (年化收益率 - 无风险利率) / 下行波动率 | 只考虑下行风险的夏普比率 |

## 5. 注意事项和最佳实践

### 5.1 数据完整性

- 确保回测期间有足够的历史数据
- 注意股票停牌、退市等情况的处理
- 检查数据质量，排除异常值

### 5.2 避免常见偏差

1. **前视偏差 (Look-ahead Bias)**:
   - 确保回测时只使用分析日期之前的数据
   - 不要使用未来信息进行决策

2. **存活偏差 (Survivorship Bias)**:
   - 包含已退市股票的历史数据
   - 使用更完整的股票池

3. **过度拟合 (Overfitting)**:
   - 避免过度优化参数
   - 使用样本外数据验证
   - 进行Walk-Forward Analysis

### 5.3 参数敏感性

- 对关键参数进行敏感性测试
- 确保策略在不同参数下表现稳定
- 避免参数过度优化

## 6. 使用示例

### 6.1 基本使用

```bash
cd stock_selector
python run_backtest.py --config backtest_config.yaml --output backtest_results
```

### 6.2 自定义股票池

编辑 `backtest_config.yaml`，设置 `stock_pool` 为具体的股票代码列表：

```yaml
stock_pool:
  - "600519"
  - "000001"
  - "600036"
```

### 6.3 使用自动获取的股票池

将 `stock_pool` 设置为空列表，系统将自动获取沪A+深A主板股票：

```yaml
stock_pool: []
```

---

**文档版本**: v1.0  
**最后更新**: 2025年  
**维护者**: FinancialTool Team
