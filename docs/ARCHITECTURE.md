# 金融工具 - 分层架构设计文档

## 概述

本文档描述金融工具项目的目标分层架构，采用渐进式重构策略，逐步实现清晰的四层架构。

## 当前架构分析

### 现有目录结构
```
FinancialTool/
├── api/              # API 层 (Presentation)
├── src/              # 核心代码
│   ├── core/         # 核心功能
│   ├── services/     # 服务层
│   └── repositories/ # 数据访问
├── stock_selector/   # 选股模块
├── watchdog/         # 盯盘模块
└── scripts/          # 脚本
```

### 优点
- API 层与业务逻辑有基本分离
- 已有服务层和数据访问层

### 不足
- 模块边界不够清晰
- 依赖关系需要规范
- 缺少明确的领域层

## 目标架构：四层架构

```
┌───────────────────────────────────────────────────────┐
│         Presentation Layer (表现层)                    │
│  - API 路由 (api/)                                     │
│  - Web UI (frontend/)                                  │
│  - 请求验证、响应格式化                                 │
├───────────────────────────────────────────────────────┤
│        Application Layer (应用层)                      │
│  - 用例编排 (Use Cases)                               │
│  - 事务管理                                            │
│  - 安全性检查                                          │
├───────────────────────────────────────────────────────┤
│         Domain Layer (领域层)                          │
│  - 领域模型 (Domain Models)                            │
│  - 领域服务 (Domain Services)                          │
│  - 策略引擎、指标计算器                                 │
├───────────────────────────────────────────────────────┤
│      Infrastructure Layer (基础设施层)                 │
│  - 数据提供者 (Data Providers)                         │
│  - 缓存 (Cache)                                        │
│  - 存储 (Storage)                                      │
│  - 外部 API 集成                                       │
└───────────────────────────────────────────────────────┘
```

## 架构原则

### 1. 依赖规则
- **外层依赖内层**：表现层 → 应用层 → 领域层 ← 基础设施层
- **内层不依赖外层**：领域层不知道 API 层的存在
- **依赖倒置**：通过接口定义，基础设施层实现领域层定义的接口

### 2. 单一职责
- 每个模块只有一个变更的理由
- 明确的模块边界

### 3. 可测试性
- 领域逻辑不依赖外部系统
- 易于编写单元测试

## 渐进式重构计划

### Phase 1: 定义接口和边界（当前）
- [x] 定义各层的职责和边界
- [ ] 创建基础设施接口
- [ ] 保持现有代码正常工作

### Phase 2: 提取领域服务
- [ ] 将业务逻辑从服务层移到领域层
- [ ] 创建领域服务
- [ ] 保持向后兼容

### Phase 3: 应用层编排
- [ ] 创建用例（Use Cases）
- [ ] 事务管理
- [ ] 权限验证

### Phase 4: 基础设施重构
- [ ] 实现领域接口
- [ ] 外部依赖隔离
- [ ] 可替换的实现

## 模块职责定义

### Presentation Layer (表现层)
**位置**: `api/`, `frontend/`

**职责**:
- 处理 HTTP 请求和响应
- 请求参数验证
- 响应格式化
- 用户认证和授权
- 路由分发

**不应该做**:
- 业务逻辑
- 直接访问数据库
- 复杂决策

### Application Layer (应用层)
**位置**: `src/application/` (新建)

**职责**:
- 用例编排（Use Cases）
- 事务管理
- 安全性检查
- 调用领域服务
- 数据转换（DTO ↔ 领域模型）

**示例用例**:
- `AnalyzeStockUseCase`
- `RunBacktestUseCase`
- `ExecuteTradeUseCase`

### Domain Layer (领域层)
**位置**: `src/domain/` (新建), `src/core/` (现有)

**职责**:
- 领域模型（实体、值对象）
- 领域服务
- 业务规则
- 策略引擎
- 指标计算

**现有模块迁移**:
- `src/core/strategy_backtest/` → 领域服务
- `src/core/indicators/` → 领域服务
- `stock_selector/` → 领域服务

### Infrastructure Layer (基础设施层)
**位置**: `src/infrastructure/` (新建)

**职责**:
- 数据提供者实现
- 缓存实现
- 数据库访问
- 外部 API 集成
- 文件存储

**接口定义在领域层**:
```python
# src/domain/repositories/stock_repository.py (接口)
class StockRepository(ABC):
    @abstractmethod
    def get_stock_data(self, code: str) -> StockData:
        pass

# src/infrastructure/repositories/sqlite_stock_repository.py (实现)
class SQLiteStockRepository(StockRepository):
    def get_stock_data(self, code: str) -> StockData:
        # SQLite 实现
        pass
```

## 依赖注入

使用依赖注入容器管理依赖：

```python
# src/application/container.py
from dependency_injector import containers, providers

class ApplicationContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    # 基础设施
    stock_repository = providers.Singleton(
        SQLiteStockRepository,
        db_path=config.db.path
    )
    
    cache_manager = providers.Singleton(
        CacheManager,
        config=config.cache
    )
    
    # 领域服务
    strategy_engine = providers.Singleton(
        StrategyEngine,
        stock_repo=stock_repository
    )
    
    # 应用服务
    analysis_service = providers.Singleton(
        AnalysisService,
        strategy_engine=strategy_engine,
        cache_manager=cache_manager
    )
```

## 示例：重构路径

### 现有代码
```python
# src/services/analysis_service.py
class AnalysisService:
    def analyze_stock(self, code: str):
        # 直接调用数据提供者
        data = get_stock_data(code)
        # 直接计算指标
        indicators = calculate_indicators(data)
        # 直接调用 AI
        analysis = call_ai_api(indicators)
        return analysis
```

### 重构后
```python
# src/domain/services/indicator_calculator.py
class IndicatorCalculator:
    def calculate(self, data: StockData) -> Indicators:
        # 纯领域逻辑
        pass

# src/application/use_cases/analyze_stock.py
class AnalyzeStockUseCase:
    def __init__(
        self,
        stock_repo: StockRepository,
        indicator_calculator: IndicatorCalculator,
        ai_service: AIService
    ):
        self.stock_repo = stock_repo
        self.indicator_calculator = indicator_calculator
        self.ai_service = ai_service
    
    def execute(self, code: str) -> AnalysisResult:
        # 编排用例
        data = self.stock_repo.get_stock_data(code)
        indicators = self.indicator_calculator.calculate(data)
        analysis = self.ai_service.analyze(indicators)
        return AnalysisResult(data=data, indicators=indicators, analysis=analysis)
```

## 迁移策略

### 1. 保持向后兼容
- 旧的 API 端点继续工作
- 逐步迁移到新架构
- 添加弃用警告

### 2. 测试驱动
- 为现有功能编写测试
- 重构后确保测试通过
- 添加新功能的测试

### 3. 增量迁移
- 每次只迁移一个模块
- 验证后再迁移下一个
- 可以随时回滚

## 总结

本架构设计提供了清晰的模块边界和依赖规则，通过渐进式重构逐步实现。关键是：
1. 保持现有功能正常工作
2. 定义清晰的接口
3. 逐步迁移到目标架构
4. 持续测试验证
