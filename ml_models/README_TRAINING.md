# 机器学习模型训练工作流使用说明

## 概述

本模块提供完整的机器学习模型训练工作流，支持从数据库或样本数据训练多种预测模型。

## 支持的模型

- XGBoost
- LightGBM
- LSTM (PyTorch)
- GRU (PyTorch)
- Ensemble (集成模型)

## 支持的预测周期

- 1天 (短线)
- 5天 (中线)
- 20天 (长线)

## 快速开始

### 1. 基础训练（使用样本数据）

```bash
python -m ml_models.train_workflow
```

### 2. 指定预测目标和模型

```bash
# 只训练1天和5天的预测模型
python -m ml_models.train_workflow --target-days 1 5

# 只训练XGBoost和LightGBM
python -m ml_models.train_workflow --models xgboost lightgbm
```

### 3. 使用真实数据库数据

```bash
python -m ml_models.train_workflow --no-sample --max-stocks 50
```

### 4. 启用超参数调优

```bash
python -m ml_models.train_workflow --tuning
```

## 完整参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--target-days` | 预测目标天数列表 | 1 5 20 |
| `--models` | 要训练的模型类型 | xgboost lightgbm lstm gru ensemble |
| `--use-sample` | 使用样本数据 | True |
| `--no-sample` | 不使用样本数据，尝试从数据库加载 | False |
| `--max-stocks` | 最大股票数量 | 与user_config.stock_count同步 |
| `--tuning` | 进行超参数调优 | False |

## Python API 使用

```python
from ml_models.train_workflow import ModelTrainer

# 创建训练器
trainer = ModelTrainer()

# 运行完整训练
reports = trainer.run(
    target_days_list=[1, 5, 20],
    model_types=['xgboost', 'lightgbm', 'lstm', 'gru', 'ensemble'],
    use_sample=True,
    max_stocks=20,
    do_tuning=False
)

# 加载已训练的模型
model = trainer.load_model('xgboost', target_days=1, use_latest=True)

# 使用模型进行预测
predictions = model.predict(X_test)
```

## 输出文件

### 模型文件

保存位置: `ml_models/models/`

文件命名格式:
- `{model_type}_{target_days}d_{date}.{ext}` - 带时间戳版本
- `{model_type}_{target_days}d_latest.{ext}` - 最新版本

扩展名:
- XGBoost: `.json`
- LightGBM/Ensemble: `.pkl`
- LSTM/GRU: `.pt`

### 训练报告

保存位置: `ml_models/reports/`

文件命名格式:
- `training_report_{target_days}d_{timestamp}.json`
- `training_report_{target_days}d_latest.json`

## 目录结构

```
ml_models/
├── models.py              # 模型定义
├── config.py              # 配置文件
├── features.py            # 特征工程
├── utils.py               # 工具函数
├── train_workflow.py      # 训练工作流 (新增)
├── test_train.py          # 测试脚本 (新增)
├── README_TRAINING.md     # 本文档 (新增)
├── models/                # 模型保存目录
│   ├── xgboost_1d_latest.json
│   ├── lightgbm_5d_latest.pkl
│   └── ...
└── reports/               # 训练报告目录
    ├── training_report_1d_latest.json
    └── ...
```

## 注意事项

1. **首次使用**: 建议先用样本数据测试: `python -m ml_models.train_workflow --max-stocks 10 --models xgboost lightgbm`

2. **股票数量同步**: `--max-stocks`参数默认值与`user_config.json`中的`stock_count`保持同步，修改user_config后会自动生效

3. **深度学习模型**: LSTM和GRU训练时间较长，建议先训练XGBoost和LightGBM

4. **超参数调优**: 启用`--tuning`会显著增加训练时间

5. **GPU加速**: 如果有可用的GPU，PyTorch模型会自动使用

6. **增量训练**: 加载已有的模型后，可以用新数据继续训练
