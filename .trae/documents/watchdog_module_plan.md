# 辅助盯盘模块 - 实现计划

## [ ] 任务 1: 创建盯盘模块目录结构
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 创建独立的 `watchdog/` 目录作为盯盘模块的根目录
  - 创建必要的子目录和初始化文件
  - 保持与 `stock_selector/` 类似的目录结构
- **Success Criteria**:
  - 目录结构创建完成
  - 所有 `__init__.py` 文件正确设置
- **Test Requirements**:
  - `programmatic` TR-1.1: 目录结构能够被 Python 正确导入
  - `human-judgement` TR-1.2: 目录结构清晰、易于维护
- **Notes**: 参考 `stock_selector/` 的目录结构

## [ ] 任务 2: 设计并实现盯盘策略数据模型
- **Priority**: P0
- **Depends On**: Task 1
- **Description**: 
  - 创建 `watchdog/base.py`，定义核心数据结构
  - 实现 `WatchdogStrategy` 基类
  - 实现 `WatchdogWatchlist` 数据类（管理自选股和策略配置）
  - 实现 `WatchdogCondition` 数据类（触发条件）
  - 实现 `WatchdogAlert` 数据类（预警信息）
  - 实现 `WatchdogDecision` 数据类（决策建议）
- **Success Criteria**:
  - 数据模型能够正确表示盯盘所需的所有信息
  - 类型注解完整且正确
- **Test Requirements**:
  - `programmatic` TR-2.1: 所有数据类能够正常实例化
  - `programmatic` TR-2.2: 基类方法能够被正确继承和重写
- **Notes**: 参考 `stock_selector/base.py` 和 `src/agent/executor.py` 的设计

## [ ] 任务 3: 实现内置盯盘策略
- **Priority**: P1
- **Depends On**: Task 2
- **Description**: 
  - 创建 `watchdog/strategies/` 目录
  - 实现基于现有交易策略的盯盘版本：
    - `ma_golden_cross_watchdog.yaml` - 均线金叉盯盘策略
    - `shrink_pullback_watchdog.yaml` - 缩量回踩盯盘策略
    - `volume_breakout_watchdog.yaml` - 放量突破盯盘策略
    - `bull_trend_watchdog.yaml` - 多头趋势盯盘策略
  - 实现策略加载器
- **Success Criteria**:
  - 内置策略能够正确定义触发条件
  - 策略加载器能够正确加载 YAML 配置
- **Test Requirements**:
  - `programmatic` TR-3.1: YAML 策略文件能够被正确解析
  - `programmatic` TR-3.2: 策略条件能够被正确应用
- **Notes**: 参考 `stock_selector/strategies/` 的实现方式

## [ ] 任务 4: 实现实时行情监控引擎
- **Priority**: P0
- **Depends On**: Task 2, Task 3
- **Description**: 
  - 创建 `watchdog/monitor.py`
  - 实现 `WatchdogMonitor` 类，负责：
    - 定期获取实时行情（利用现有 `data_provider`）
    - 检查策略触发条件
    - 生成预警信息和决策建议
    - 管理监控状态
  - 集成现有的 `DataFetcherManager` 获取实时数据
  - 实现频率控制和市场开盘时间检测
- **Success Criteria**:
  - 监控引擎能够定期获取实时数据
  - 能够正确检测策略触发条件
  - 能够生成预警和决策建议
- **Test Requirements**:
  - `programmatic` TR-4.1: 监控引擎能够正常启动和停止
  - `programmatic` TR-4.2: 能够正确获取和处理实时数据
- **Notes**: 参考 `src/core/pipeline.py` 的数据获取方式和 `src/scheduler.py` 的调度方式

## [ ] 任务 5: 实现盯盘配置管理器
- **Priority**: P0
- **Depends On**: Task 2
- **Description**: 
  - 创建 `watchdog/config.py`
  - 实现 `WatchdogConfig` 类，负责：
    - 管理自选股和策略配置
    - 持久化配置到数据库或文件
    - 支持配置的增删改查
  - 集成现有的 `src/storage.py` 进行数据持久化
- **Success Criteria**:
  - 配置能够正确保存和加载
  - 配置变更能够及时生效
- **Test Requirements**:
  - `programmatic` TR-5.1: 配置能够正确保存到存储
  - `programmatic` TR-5.2: 配置能够从存储正确加载
- **Notes**: 参考 `stock_selector/config.py` 的实现方式

## [ ] 任务 6: 实现预警通知集成
- **Priority**: P1
- **Depends On**: Task 4, Task 5
- **Description**: 
  - 创建 `watchdog/notifier.py`
  - 实现 `WatchdogNotifier` 类，负责：
    - 接收预警信息
    - 格式化预警消息
    - 调用现有的 `NotificationService` 发送通知
  - 支持多种通知渠道（复用现有通知系统）
- **Success Criteria**:
  - 预警能够正确发送到已配置的通知渠道
  - 预警消息格式清晰、信息完整
- **Test Requirements**:
  - `programmatic` TR-6.1: 预警通知能够正确调用 `NotificationService`
  - `human-judgement` TR-6.2: 预警消息内容完整、格式清晰
- **Notes**: 复用 `src/notification.py` 的通知系统

## [ ] 任务 7: 实现盯盘服务主类
- **Priority**: P0
- **Depends On**: Task 4, Task 5, Task 6
- **Description**: 
  - 创建 `watchdog/service.py`
  - 实现 `WatchdogService` 类，作为盯盘模块的入口点，负责：
    - 协调监控引擎、配置管理器和通知器
    - 提供启动/停止/状态查询等接口
    - 管理监控生命周期
  - 创建 `watchdog/__init__.py` 导出公共 API
- **Success Criteria**:
  - 服务能够正常启动和停止
  - 能够正确协调各子模块工作
- **Test Requirements**:
  - `programmatic` TR-7.1: 服务能够正常启动和停止
  - `programmatic` TR-7.2: 公共 API 能够正确导出和调用
- **Notes**: 参考 `stock_selector/service.py` 的设计

## [ ] 任务 8: 实现 API 端点
- **Priority**: P1
- **Depends On**: Task 7
- **Description**: 
  - 创建 `api/v1/endpoints/watchdog.py`
  - 创建 `api/v1/schemas/watchdog.py`
  - 实现以下 API 端点：
    - GET /api/v1/watchdog/watchlist - 获取自选股列表
    - POST /api/v1/watchdog/watchlist - 添加自选股
    - DELETE /api/v1/watchdog/watchlist/{code} - 删除自选股
    - GET /api/v1/watchdog/strategies - 获取可用策略列表
    - POST /api/v1/watchdog/strategies/{code} - 配置股票策略
    - POST /api/v1/watchdog/start - 启动盯盘
    - POST /api/v1/watchdog/stop - 停止盯盘
    - GET /api/v1/watchdog/status - 获取盯盘状态
    - GET /api/v1/watchdog/alerts - 获取预警历史
- **Success Criteria**:
  - API 端点能够正常响应
  - 数据验证完整
- **Test Requirements**:
  - `programmatic` TR-8.1: API 端点能够正确路由和响应
  - `programmatic` TR-8.2: 输入数据验证正确
- **Notes**: 参考 `api/v1/endpoints/stock_selector.py` 的实现方式

## [ ] 任务 9: 创建示例和文档
- **Priority**: P2
- **Depends On**: Task 1-8
- **Description**: 
  - 创建 `watchdog/examples/` 目录
  - 创建示例配置文件
  - 创建简单的使用示例脚本
  - 更新项目文档（可选）
- **Success Criteria**:
  - 示例文件能够正常运行
  - 文档清晰易懂
- **Test Requirements**:
  - `programmatic` TR-9.1: 示例脚本能够正常运行
  - `human-judgement` TR-9.2: 文档清晰易懂
- **Notes**: 参考 `stock_selector/examples/` 的内容
