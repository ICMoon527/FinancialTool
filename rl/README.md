# 强化学习分时做T训练模块

## 快速开始

```bash
# 1. 冒烟测试（模拟数据，验证环境）
python -m rl.tests.test_dqn_smoke

# 2. 使用真实数据训练（需先确保数据库有分时数据）
python -m rl.scripts.train_dqn
```

---

## 使用方式

### 方式一：脚本训练（推荐，无需前端）

```python
# rl/scripts/train_dqn.py
from rl.config import RLConfig
from rl.algorithms.dqn import DQNModel
from rl.data.dataset import IntradayDataset
from rl.training.trainer import RLTrainer
from rl.training.callbacks import ProgressCallback
from src.storage import get_db

# 加载配置
config = RLConfig.from_env()
# 覆盖参数（可选）
config.training_episodes = 500
config.batch_size = 128

# 加载真实数据
db = get_db()
dataset = IntradayDataset(config, db)
dataset.load()
print(f"训练集: {len(dataset.train_samples)}, 验证集: {len(dataset.val_samples)}")

# 创建模型
model = DQNModel(config)

# 训练
progress_store = {}
trainer = RLTrainer(
    config=config,
    model=model,
    dataset=dataset,
    callbacks=[ProgressCallback(progress_store)],
)
metrics = trainer.train()

# 查看结果
print(f"最终验证 Sharpe: {max(metrics.val_sharpe_ratios):.4f}")
```

### 方式二：API 调用（需要先启动后端）

```bash
# 启动训练
curl -X POST http://localhost:8000/api/v1/rl/train \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "dqn", "episodes": 500}'
# 返回: {"task_id": "xxx", "status": "pending", "message": "训练任务已创建"}

# 查询进度
curl http://localhost:8000/api/v1/rl/train/{task_id}/progress

# 获取模型列表
curl http://localhost:8000/api/v1/rl/models

# 评估模型
curl -X POST http://localhost:8000/api/v1/rl/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model_id": "xxx"}'

# 单日回放
curl http://localhost:8000/api/v1/rl/evaluate/{model_id}/daily/000001/2025-06-15
```

### 方式三：加载已有模型推理

```python
from rl.config import RLConfig
from rl.algorithms.dqn import DQNModel
from rl.environment import T0Environment
from src.storage import get_db

config = RLConfig.from_env()
model = DQNModel(config)
model.load("rl/models/dqn_best_20260630_175641/model.pt")

# 用真实数据跑一个交易日
db = get_db()
klines = db.load_intraday_klines("000001", date(2025, 6, 15))
prev_klines = db.load_prev_day_klines("000001", date(2025, 6, 15))

env = T0Environment(config)
state = env.reset(
    {"klines": klines, "stock_code": "000001", "date": date(2025, 6, 15)},
    prev_day_klines=prev_klines,
)

done = False
while not done:
    action = model.predict(state, deterministic=True)
    state, reward, done, _ = env.step(action)
```

---

## 配置项说明

通过 `.env` 文件配置，支持以下参数（括号内为默认值）：

| 分类 | 变量 | 默认值 | 说明 |
|------|------|--------|------|
| 通用 | `RL_ENABLED` | `false` | 是否启用 |
| 通用 | `RL_DEFAULT_ALGORITHM` | `dqn` | 算法 (dqn/ppo) |
| 训练 | `RL_TRAINING_EPISODES` | `1000` | 训练轮数 |
| 训练 | `RL_BATCH_SIZE` | `64` | 批次大小 |
| 训练 | `RL_LEARNING_RATE` | `0.001` | 学习率 |
| 训练 | `RL_GAMMA` | `0.99` | 折扣因子 |
| 训练 | `RL_WARMUP_STEPS` | `20` | 预热步数 |
| 训练 | `RL_VALIDATION_FREQ` | `50` | 验证频率 |
| 训练 | `RL_EARLY_STOPPING_PATIENCE` | `10` | 早停耐心值 |
| 数据 | `RL_TRAIN_DATA_DAYS` | `60` | 训练数据天数 |
| 数据 | `RL_VALIDATION_SPLIT` | `0.2` | 验证集比例 |
| 奖励 | `RL_DENSE_REWARD_SCALE` | `20` | 密集奖励缩放 |
| 奖励 | `RL_REWARD_CLIP` | `5.0` | 奖励裁剪范围 |
| 交易成本 | `RL_COMMISSION_RATE` | `0.001` | 佣金费率 |
| 交易成本 | `RL_SLIPPAGE_RATE` | `0.001` | 滑点费率 |
| DQN | `RL_EPSILON_START` | `1.0` | 探索率初始值 |
| DQN | `RL_EPSILON_END` | `0.01` | 探索率终值 |
| DQN | `RL_EPSILON_DECAY` | `0.995` | 探索率衰减 |
| DQN | `RL_REPLAY_BUFFER_SIZE` | `10000` | 经验回放容量 |
| DQN | `RL_TARGET_UPDATE_FREQ` | `100` | 目标网络更新频率 |
| DQN | `RL_DQN_DOUBLE` | `true` | Double DQN |
| DQN | `RL_DQN_DUELING` | `true` | Dueling DQN |
| DQN | `RL_DQN_HIDDEN_SIZES` | `256,128,64` | 隐藏层大小 |
| 存储 | `RL_MODEL_DIR` | `rl/models` | 模型存储目录 |
| 存储 | `RL_SAVE_BEST_ONLY` | `true` | 仅保存最佳模型 |

---

## 模型输出

训练完成后，`rl/models/` 下生成：

```
rl/models/
└── dqn_best_20260630_175641/
    ├── model.pt      # 模型权重（q_network + target_network + optimizer + epsilon）
    ├── config.json    # 训练配置
    └── metrics.json   # 训练指标（reward曲线、loss、Sharpe等）
```

---

## 目录结构

```
rl/
├── config.py              # RLConfig 配置类
├── environment.py         # T0Environment 交易环境
├── networks.py            # DQN/PPO 神经网络
├── algorithms/
│   ├── base.py            # AbstractRLModel 基类
│   └── dqn.py             # DQNModel + ReplayBuffer
├── training/
│   ├── trainer.py         # RLTrainer 训练编排
│   └── callbacks.py       # 回调（进度/checkpoint）
├── evaluation/
│   └── evaluator.py       # RLEvaluator 评估器
├── data/
│   └── dataset.py         # IntradayDataset 数据集
├── tests/
│   └── test_dqn_smoke.py  # 冒烟测试
├── models/                # 训练产物（自动生成）
└── README.md              # 本文件
```

---

## 环境说明

- **动作空间**：0=HOLD, 1=BUY, 2=SELL
- **底仓管理**：初始 3 份底仓，SELL 卖出底仓，BUY 买回
- **T+1 规则**：当日买入不可当日卖出
- **预热期**：前 `warmup_steps` 步强制 HOLD，reward=0
- **状态维度**：50 维（价格、量能、MACD、RSI、KDJ、MFI、仓位、时间、预热标志）
- **GPU 支持**：自动检测 CUDA，有 GPU 则自动使用