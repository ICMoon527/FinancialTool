# 强化学习分时做T训练模块

## 快速开始

```bash
# 1. 冒烟测试（模拟数据，验证环境）
python -m rl.tests.test_dqn_smoke

# 2. 使用真实数据训练（需先确保数据库有分时数据）
python -m rl.scripts.train_dqn
```

> 建议使用项目虚拟环境：`stock_venv/Scripts/python.exe -m rl.tests.test_dqn_smoke`

---

## 冒烟测试 ✅ 已通过（2026-08-27）

使用 `stock_venv` 环境（torch 2.7.1+cu118, RTX 4090）验证结果：

| 测试项 | 结果 | 关键输出 |
|--------|------|----------|
| 1. 配置加载 | ✅ | `RLConfig` 正常，`state_dim=50`, `action_dim=3` |
| 2. 环境测试 | ✅ | `reset()` 返回 50 维，BUY/SELL 动作有效 |
| 3. 网络测试 | ✅ | DQN / Dueling DQN 前向传播 shape `[4, 3]` |
| 4. DQN 模型 | ✅ | `train_step` loss=1.41，save/load 正常 |
| 5. 完整训练 | ✅ | 50 episodes，设备 cuda，总耗时约 3.5 分钟 |

训练过程中 avg_reward 从 -16.36（E10）升至 31.01（E40），学习趋势正常。
模拟数据为随机游走，负 Sharpe 属正常现象（验证的是管道而非策略质量）。

---

## 真实数据训练命令

```bash
# 默认配置训练（参数读取 .env）
python -m rl.scripts.train_dqn

# 命令行覆盖参数
python -m rl.scripts.train_dqn --episodes 500          # 训练轮数
python -m rl.scripts.train_dqn --batch-size 128        # 批次大小
python -m rl.scripts.train_dqn --lr 0.0005             # 学习率
python -m rl.scripts.train_dqn --stock 000001,600519   # 仅用指定股票
python -m rl.scripts.train_dqn --max-samples 10000     # 样本总数上限（默认5000，0=不限制）
python -m rl.scripts.train_dqn --rebuild               # 强制重建元数据缓存
python -m rl.scripts.train_dqn --no-gpu                # 强制 CPU
```

**真实数据验证记录（2026-08-28）**：27.3 万股票交易日样本元数据构建完成，
`--episodes 20 --max-samples 200` 小规模训练跑通（约 50 秒，GPU），
avg_reward 从 -15.79（E10）升至 -2.32（E20），模型正常保存。

### 断点续训与日志

```bash
# 断点续训（自动从 rl/models/dqn_latest 恢复：权重/优化器/epsilon/replay buffer/指标历史）
python -m rl.scripts.train_dqn --resume --episodes 500

# 指定 checkpoint 目录续训
python -m rl.scripts.train_dqn --resume --resume-path rl/models/dqn_best_xxx --episodes 500

# 自定义 latest checkpoint 保存频率（默认每 50 集覆盖保存一次）
python -m rl.scripts.train_dqn --save-freq 20

# 自定义日志目录（默认 rl/models/logs/）
python -m rl.scripts.train_dqn --log-dir rl/logs
```

**Checkpoint 说明**：
- `dqn_latest/`：固定目录，每 `save_freq` 集覆盖写入，用于断点续训；训练结束时也更新为该次最终状态
- `dqn_best/`：固定目录，验证 Sharpe 创新高时覆盖写入，始终保留历史最优
- `trainer_state.json`：续训状态（episode 进度、最佳 Sharpe、早停计数、run_id）

**日志说明**（默认位于 `rl/models/logs/`）：
- `train_YYYYMMDD_HHMMSS.log`：运行日志（控制台 + 文件双写）
- `train_log_<run_id>.csv`：逐集指标（reward/loss/td_error/epsilon/验证指标/耗时），
  每集立即落盘（崩溃安全），断点续训时同一 run_id 续写同一 CSV

**续训验证记录（2026-08-28）**：10 集训练 → `--resume` 续训至 15 集，
replay buffer（2380 条）与指标历史完整恢复，CSV 日志连续。

### 超大数据库性能说明（重要）

本项目数据库约 14.7GB（27 万+ 股票日样本，约 9000 万行分时K线），
且位于 **USB 外接机械硬盘**上。已做针对性优化：

1. **惰性加载**：内存只存 (股票, 日期) 元数据，K线训练时按需查询（单日索引查询）
2. **元数据缓存**：首次启动构建索引约 15-20 分钟（机械盘全索引扫描），
   结果保存至 `rl/data/_meta_cache.pkl`，后续启动毫秒级加载
3. **样本上限**：`--max-samples`（默认 5000）随机下采样，防止验证集拖慢训练
4. **验证集抽样**：每轮验证最多随机抽 50 天

建议：若训练速度不理想，可考虑将 `data/stock_analysis.db` 迁移到 NVMe SSD。

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