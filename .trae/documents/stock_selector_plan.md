# Stock Selector - 选股功能实现计划

## 项目概述
实现一个可扩展的选股系统，支持自然语言策略（兼容现有 `strategies/` 目录）和 Python 代码策略两种模式，新增 WebUI 标签"选股"。

## 架构设计

### 目录结构
```
e:\Code\daily_stock_analysis\
├── stock_selector/              # 新模块 - 选股系统
│   ├── __init__.py
│   ├── base.py                  # 选股策略基类
│   ├── manager.py               # 策略管理器
│   ├── service.py               # 选股服务层
│   ├── strategies/               # 选股策略目录
│   │   ├── __init__.py
│   │   ├── nl_strategy_loader.py    # 自然语言策略加载器
│   │   └── python_strategy_loader.py # Python代码策略加载器
│   └── examples/               # 示例策略
│       ├── short_term_strategy.yaml  # 下午两点半短线策略（自然语言）
│       └── short_term_strategy.py    # 下午两点半短线策略（Python代码）
├── api/
│   └── v1/
│       └── endpoints/
│           └── stock_selector.py  # 选股API端点
└── apps/dsa-web/
    └── src/
        ├── pages/
        │   └── StockSelectorPage.tsx  # 选股页面
        ├── api/
        │   └── stockSelector.ts      # 选股API客户端
        └── types/
            └── stockSelector.ts       # 选股类型定义
```

---

## 示例策略说明

### 下午两点半短线策略

**条件：**
1. 下午两点半涨幅 3% 到 5%
2. 剔除量比小于 1 的
3. 保留换手率 5% 到 10%
4. 保留流通市值在 50 亿到 200 亿
5. 留下成交量持续放大的，最好是台阶式
6. 查看 K 线形态，保留多头向上发散
7. 叠加上证指数的分时图看，个股要全天在上证指数的上方
8. 如果筛选出来的个股两点半之后创了新高，就是目标个股
9. 当回落到均线附近的时候，不跌破就是最好的入场时机
10. 做短线做好止盈止损，快准狠，快进快出
11. 盈利之后如果没有按照预期发展也可以直接出场

**输出要求：**
- 选出最符合所有策略的前五支标的
- 标明每支股票符合哪些策略，不符合哪些策略

---

## 任务列表

### [ ] Task 1: 创建 stock_selector 模块目录结构和基类
- **Priority**: P0
- **Depends On**: None
- **Description**:
  - 创建 `stock_selector/` 目录结构
  - 定义 `StockSelectorStrategy` 基类（所有策略的统一接口）
  - 定义 `StockCandidate` 数据类（选股结果结构，包含策略匹配度信息）
  - 定义 `StrategyMatch` 数据类（单策略匹配结果）
  - 定义策略元数据接口
  - 定义策略匹配评分机制（用于综合排名）
  - 创建示例策略文件：
    - `short_term_strategy.yaml`（自然语言版本）
    - `short_term_strategy.py`（Python 代码版本）
- **Success Criteria**:
  - 目录结构创建成功
  - 基类定义完整，包含统一的 `select()` 方法
  - 数据类定义符合 TypeScript 类型标准，包含策略匹配度标记
  - 示例策略文件创建成功
- **Test Requirements**:
  - `programmatic` TR-1.1: 模块可以正常导入，无语法错误
  - `human-judgement` TR-1.2: 基类设计符合现有代码风格，具有可扩展性

---

### [ ] Task 2: 实现自然语言策略加载器
- **Priority**: P0
- **Depends On**: Task 1
- **Description**:
  - 创建 `nl_strategy_loader.py`
  - 复用现有 `strategies/` 目录的 YAML 策略
  - 适配自然语言策略到选股接口
  - 通过 LLM 解析策略描述并执行选股
  - 支持策略匹配度标记（符合/不符合）
- **Success Criteria**:
  - 能够从 `strategies/` 加载所有 YAML 文件
  - 能够加载 `short_term_strategy.yaml` 示例策略
  - 将自然语言策略转换为可执行的选股逻辑
- **Test Requirements**:
  - `programmatic` TR-2.1: 能成功加载 bull_trend.yaml 等现有策略
  - `programmatic` TR-2.2: 能成功加载 short_term_strategy.yaml 示例策略
  - `human-judgement` TR-2.3: 加载器代码清晰，错误处理完善

---

### [ ] Task 3: 实现 Python 代码策略框架
- **Priority**: P0
- **Depends On**: Task 1
- **Description**:
  - 创建 `python_strategy_loader.py`
  - 实现策略装饰器/注册机制
  - 支持热加载和动态发现
  - 提供策略开发模板
  - 实现 `short_term_strategy.py` 示例策略（下午两点半策略）
- **Success Criteria**:
  - 可以自动发现和加载 Python 策略文件
  - 策略可以访问数据获取接口
  - 策略返回匹配度信息（符合/不符合）
- **Test Requirements**:
  - `programmatic` TR-3.1: 能成功注册和发现示例策略
  - `programmatic` TR-3.2: short_term_strategy.py 能正常执行
  - `human-judgement` TR-3.3: 框架设计易于扩展，文档友好

---

### [ ] Task 4: 创建策略管理器
- **Priority**: P0
- **Depends On**: Tasks 1, 2, 3
- **Description**:
  - 创建 `manager.py`
  - 管理策略的加载、激活、停用
  - 支持策略组合（同时运行多个策略）
  - 提供策略元数据查询接口
  - 实现策略匹配度综合评分和排序
  - 支持取 Top N 结果（默认 Top 5）
- **Success Criteria**:
  - 可以统一管理自然语言和 Python 策略
  - 支持策略状态管理（激活/停用）
  - 能综合多个策略的匹配度进行排序
- **Test Requirements**:
  - `programmatic` TR-4.1: 能同时激活多个策略并执行选股
  - `programmatic` TR-4.2: 能正确返回 Top 5 结果
  - `human-judgement` TR-4.3: 管理器设计清晰，易于理解和维护

---

### [ ] Task 5: 创建选股服务层
- **Priority**: P0
- **Depends On**: Task 4
- **Description**:
  - 创建 `service.py`
  - 实现选股执行逻辑
  - 支持并发选股和限流
  - 结果缓存和持久化（可选）
  - 实现策略匹配度标记（每支股票符合哪些策略）
  - 实现综合排序和 Top 5 筛选
- **Success Criteria**:
  - 可以执行单个或多个策略的选股
  - 结果格式统一，便于前端展示
  - 结果包含策略匹配度信息
- **Test Requirements**:
  - `programmatic` TR-5.1: 选股服务能正常执行并返回结果
  - `programmatic` TR-5.2: 结果包含策略匹配度标记
  - `human-judgement` TR-5.3: 服务层设计合理，性能可接受

---

### [ ] Task 6: 创建 API 端点
- **Priority**: P1
- **Depends On**: Task 5
- **Description**:
  - 创建 `api/v1/endpoints/stock_selector.py`
  - 实现策略列表查询接口
  - 实现选股执行接口
  - 实现结果查询接口
  - 注册到 `api/v1/router.py`
  - API 响应包含策略匹配度和 Top 5 结果
- **Success Criteria**:
  - API 端点完整，符合 OpenAPI 规范
  - 可以通过 API 执行选股并获取结果
  - API 响应格式包含策略匹配度信息
- **Test Requirements**:
  - `programmatic` TR-6.1: 所有 API 端点能正常响应
  - `programmatic` TR-6.2: API 响应包含策略匹配度信息
  - `human-judgement` TR-6.3: API 设计符合 RESTful 规范，文档清晰

---

### [ ] Task 7: 添加 TypeScript 类型定义
- **Priority**: P1
- **Depends On**: Task 6
- **Description**:
  - 创建 `apps/dsa-web/src/types/stockSelector.ts`
  - 定义策略元数据类型
  - 定义选股请求/响应类型
  - 定义选股结果类型（包含策略匹配度）
  - 定义 `StrategyMatch` 类型（单策略匹配结果）
- **Success Criteria**:
  - 类型定义完整，与后端 API 一致
  - 类型安全，无 any 类型
- **Test Requirements**:
  - `programmatic` TR-7.1: TypeScript 编译无错误
  - `human-judgement` TR-7.2: 类型定义清晰，易于使用

---

### [ ] Task 8: 创建选股 API 客户端
- **Priority**: P1
- **Depends On**: Task 7
- **Description**:
  - 创建 `apps/dsa-web/src/api/stockSelector.ts`
  - 实现策略列表查询
  - 实现选股执行
  - 实现结果查询
- **Success Criteria**:
  - API 客户端完整，覆盖所有端点
  - 错误处理完善
- **Test Requirements**:
  - `programmatic` TR-8.1: API 客户端能正常调用后端
  - `human-judgement` TR-8.2: 客户端代码风格与现有 API 一致

---

### [ ] Task 9: 创建选股页面 UI
- **Priority**: P1
- **Depends On**: Task 8
- **Description**:
  - 创建 `apps/dsa-web/src/pages/StockSelectorPage.tsx`
  - 策略选择器（多选，支持自然语言和 Python 策略）
  - 选股执行按钮和状态显示
  - 选股结果列表展示（表格/卡片）
  - 显示 Top 5 结果
  - 每支股票显示：
    - 基本信息（代码、名称、价格等）
    - 符合的策略列表（绿色标记）
    - 不符合的策略列表（红色/灰色标记）
    - 综合评分
  - 结果筛选和排序
- **Success Criteria**:
  - UI 功能完整，用户体验良好
  - 样式与现有页面一致
  - 能正确显示策略匹配度
- **Test Requirements**:
  - `programmatic` TR-9.1: 页面能正常加载和交互
  - `programmatic` TR-9.2: 能正确显示 Top 5 结果和策略匹配度
  - `human-judgement` TR-9.3: UI 设计美观，操作流畅

---

### [ ] Task 10: 更新 App.tsx 添加导航和路由
- **Priority**: P1
- **Depends On**: Task 9
- **Description**:
  - 添加选股图标组件
  - 更新 `NAV_ITEMS` 添加"选股"导航项
  - 更新 `Routes` 添加选股页面路由
- **Success Criteria**:
  - 导航栏显示"选股"图标
  - 可以正常访问选股页面
- **Test Requirements**:
  - `programmatic` TR-10.1: 导航和路由能正常工作
  - `human-judgement` TR-10.2: 导航设计与现有风格一致

---

## 兼容性说明

- 所有新建代码独立于 `stock_selector/` 目录
- 最小化对现有代码的修改
- 复用现有数据获取模块（`data_provider/`）
- 复用现有 Agent 策略加载机制

## 代码风格约定

- Python: 遵循 `black` + `isort` + `flake8`，行宽 120
- TypeScript: 遵循现有代码风格，使用 `eslint`
- 注释: 新增代码注释使用英文
- 提交: commit message 使用英文

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 性能问题（全市场选股） | 中 | 高 | 分批处理，支持异步，添加进度指示 |
| 策略冲突 | 低 | 中 | 策略结果取交集/并集，支持权重配置 |
| 与现有代码耦合 | 低 | 低 | 新模块独立，通过接口交互 |
