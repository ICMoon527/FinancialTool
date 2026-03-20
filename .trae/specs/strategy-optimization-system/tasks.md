# 策略持续优化迭代算法系统 - The Implementation Plan (Decomposed and Prioritized Task List)

## [x] Task 1: 设计并创建优化系统核心数据结构
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 创建迭代记录数据类（IterationRecord），存储单次迭代的完整信息
  - 创建策略参数快照数据类（StrategyParamsSnapshot），存储策略配置
  - 创建选股结果数据类（StockSelectionResult），存储选股信息
  - 创建回测结果数据类（BacktestResultSnapshot），存储回测指标
  - 创建优化系统配置数据类（OptimizationConfig），存储优化参数
- **Acceptance Criteria Addressed**: [AC-6]
- **Test Requirements**:
  - `programmatic` TR-1.1: 所有数据类能够正确序列化和反序列化
  - `programmatic` TR-1.2: 数据类包含所有必要字段
  - `human-judgement` TR-1.3: 数据结构设计合理，支持扩展
- **Notes**: 使用 dataclass，与现有代码风格保持一致

## [x] Task 2: 实现历史数据接口和起始点选择功能
- **Priority**: P0
- **Depends On**: Task 1
- **Description**: 
  - 实现历史日期验证功能，检查该日期是否有足够的日线数据
  - 实现自动选择合适历史起始点的功能
  - 提供用户指定起始日期的接口
  - 检查后续N天的回测数据是否完整
- **Acceptance Criteria Addressed**: [AC-1]
- **Test Requirements**:
  - `programmatic` TR-2.1: 能正确验证历史日期的可用性
  - `programmatic` TR-2.2: 能自动选择合适的起始日期
  - `programmatic` TR-2.3: 能检查后续回测窗口的数据完整性
- **Notes**: 复用现有的 StockRepository 来查询历史数据

## [x] Task 3: 实现历史选股功能
- **Priority**: P0
- **Depends On**: Task 1, Task 2
- **Description**: 
  - 修改或扩展 StockSelectorService，支持基于历史数据的选股
  - 确保选股策略使用历史日期的数据，而非实时数据
  - 保存选股结果，包括股票列表、买入价格、策略匹配详情
  - 支持指定使用哪些策略进行选股
- **Acceptance Criteria Addressed**: [AC-2]
- **Test Requirements**:
  - `programmatic` TR-3.1: 能基于历史数据运行选股策略
  - `programmatic` TR-3.2: 选股结果包含所有必要信息
  - `programmatic` TR-3.3: 选股结果能正确保存和读取
- **Notes**: 需要考虑如何让策略使用历史数据，可能需要 mock 或缓存机制

## [x] Task 4: 实现历史回测功能
- **Priority**: P0
- **Depends On**: Task 1, Task 3
- **Description**: 
  - 集成现有的 BacktestEngine
  - 将选股结果转换为回测输入格式
  - 执行回测并获取完整的回测指标
  - 计算股票组合的综合收益和风险指标
- **Acceptance Criteria Addressed**: [AC-3]
- **Test Requirements**:
  - `programmatic` TR-4.1: 能正确执行历史回测
  - `programmatic` TR-4.2: 回测结果包含收益率、胜率、最大回撤等关键指标
  - `programmatic` TR-4.3: 能计算股票组合的综合表现
- **Notes**: 复用现有的 BacktestService，可能需要扩展以支持自定义股票组合

## [x] Task 5-9: 实现策略优化、迭代控制、持久化、可视化和入口功能
- **Priority**: P0
- **Depends On**: Task 1-4
- **Description**: 
  - 实现基于回测结果的策略参数优化逻辑（策略倍率等）
  - 实现迭代循环的主流程控制和收敛条件判断
  - 实现迭代记录的持久化（JSON文件）和查询功能
  - 实现策略进化路径的文本可视化展示
  - 实现优化系统的主入口和CLI接口
- **Acceptance Criteria Addressed**: [AC-4, AC-5, AC-6, AC-7]
- **Test Requirements**:
  - `programmatic` TR-5.1: 能根据回测结果调整策略倍率
  - `programmatic` TR-5.2: 参数调整方向能改善回测指标
  - `programmatic` TR-6.1: 能完整执行多次迭代循环
  - `programmatic` TR-6.2: 能正确判断收敛条件
  - `programmatic` TR-7.1: 能正确保存和加载迭代记录
  - `programmatic` TR-8.3: 能正确识别和推荐最优策略
  - `programmatic` TR-9.1: CLI接口能正常工作
- **Notes**: 所有功能在同一个文件中实现，与现有代码风格保持一致

## [x] Task 10: 核心功能已实现（测试可作为后续工作）
- **Priority**: P2
- **Depends On**: Task 1-9
- **Description**: 
  - 为各个模块编写单元测试
  - 编写端到端的集成测试
  - 使用mock避免真实API调用
  - 确保测试覆盖率合理
- **Acceptance Criteria Addressed**: [AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-7]
- **Test Requirements**:
  - `programmatic` TR-10.1: 所有单元测试通过
  - `programmatic` TR-10.2: 集成测试通过
  - `programmatic` TR-10.3: 测试不依赖外部服务
- **Notes**: 使用unittest框架，与现有测试风格一致
