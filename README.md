<div align="center">

# 金融工具 — AI 股票智能分析系统

**AI-powered stock analysis system for A-shares / Hong Kong / US stocks**

基于 AI 大模型的 A股/港股/美股自选股智能分析系统，集分析、选股、监控、回测于一体，通过多渠道推送「决策仪表盘」。

[功能特性](#-功能特性) · [技术栈](#-技术栈与数据来源) · [项目结构](#-项目结构) · [快速开始](#-快速开始) · [配置说明](#-配置说明) · [免责声明](#-免责声明)

</div>

## 📖 项目说明

本项目从 [daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) 处借鉴了大量代码，在此基础上进行开发和优化，新增了 **选股系统、盘中实时监控、WebUI 管理界面、Electron 桌面应用、Docker 容器化部署、消息平台 Bot 交互** 等大量功能。

## ✨ 功能特性

| 模块       | 功能         | 说明                                                                                     |
| ---------- | ------------ | ---------------------------------------------------------------------------------------- |
| AI 分析    | 决策仪表盘   | 一句话核心结论 + 精确买卖点位 + 操作检查清单                                             |
| 分析       | 多维度分析   | 技术面（盘中实时 MA / 多头排列）+ 筹码分布 + 舆情情报 + 实时行情                         |
| 市场       | 全球市场     | 支持 A股、港股、美股及美股指数（SPX、DJI、IXIC 等）                                      |
| 策略       | 市场策略系统 | 内置 A股「三段式复盘策略」与美股「Regime Strategy」                                      |
| 复盘       | 大盘复盘     | 每日市场概览、板块涨跌；支持 cn(A股) / us(美股) / both(两者) 切换                        |
| 智能导入   | 多源导入     | 支持图片、CSV/Excel 文件、剪贴板粘贴；Vision LLM 提取代码+名称                           |
| 回测       | AI 回测验证  | 自动评估历史分析准确率，方向胜率、止盈止损命中率，支持贝叶斯参数优化                     |
| Agent 问股 | 策略对话     | 多轮策略问答，支持均线金叉 / 缠论 / 波浪等 11 种内置策略                                 |
| 选股       | 智能选股     | 六维选股系统，支持自定义 Python / YAML 策略，批量评估与回测                              |
| 盘中监控   | 实时看盘     | 分时页面实时行情轮询，T+0 策略支持，风险自动管理                                         |
| 推送       | 多渠道通知   | 企业微信、飞书、Telegram、钉钉、邮件、Pushover、PushPlus、Server酱、Discord、自定义 Webhook |
| 定时任务   | 本地调度     | 内置 scheduler，支持每日定时自动运行、交易日历智能跳过                                   |
| WebUI      | 管理界面     | React 19 + TypeScript + Tailwind CSS 4 + Vite，完整单页应用                              |
| 桌面应用   | 跨平台桌面端 | Electron 桌面应用，Windows / macOS 支持                                                  |
| Docker     | 容器化部署   | Docker + Docker Compose 一键部署，支持 analyzer 和 server 两种模式                       |
| Bot        | 消息交互     | 钉钉、飞书 Stream 模式 Bot，支持 `/分析` `/批量` `/问股` `/市场` 等命令                  |

## 🔧 技术栈与数据来源

| 类型       | 支持                                                                                 |
| ---------- | ------------------------------------------------------------------------------------ |
| AI 模型    | LiteLLM 统一调用层 — Gemini、Claude、OpenAI、DeepSeek、通义千问、Kimi、MiniMax 等    |
| 行情数据   | AkShare、Tushare、Pytdx（通达信）、Baostock、YFinance、efinance（东方财富）           |
| 新闻搜索   | Tavily、SerpAPI、Brave                                                                |
| 后端框架   | FastAPI + SQLAlchemy + Redis + SQLite                                                 |
| 前端框架   | React 19 + TypeScript + Tailwind CSS 4 + Vite + ECharts                               |
| 桌面应用   | Electron 31                                                                           |
| 量化分析   | quantstats + scikit-optimize（贝叶斯优化）                                           |
| 指标库     | 内置 150+ 技术指标（MA / MACD / KDJ / RSI / 布林带 / 缠论 / 波浪等）                 |
| 代码质量   | black + isort + flake8 + pyright                                                      |

## 📁 项目结构

```
FinancialTool/
├── main.py                    # 主调度入口（分析管线 + 定时任务 + Web 服务）
├── server.py                  # FastAPI 服务入口
├── webui.py                   # WebUI 启动脚本
├── watchdog_main.py           # 盘中实时监控入口
├── api/                       # FastAPI API 层
│   ├── middlewares/           # 认证、错误处理中间件
│   └── v1/
│       ├── endpoints/         # 接口端点（分析、回测、选股、分时等）
│       └── schemas/           # Pydantic 请求/响应模型
├── apps/
│   ├── dsa-web/               # React Web 前端（TypeScript + Vite）
│   └── dsa-desktop/           # Electron 桌面应用
├── bot/                       # 消息平台 Bot
│   ├── commands/              # Bot 命令实现
│   └── platforms/             # 平台适配（钉钉、飞书 Stream、Discord）
├── data_provider/             # 多源数据提供层（AkShare / Tushare / Pytdx / efinance 等）
├── docker/                    # Docker 部署配置（Dockerfile + docker-compose.yml）
├── docs/                      # 文档与配置截图
├── indicators/                # 技术指标库（150+ 独立指标模块）
├── patch/                     # 数据源补丁（东财反爬等）
├── scripts/                   # 构建 / 测试 / 迁移脚本
├── src/
│   ├── agent/                 # Agent 策略对话系统
│   │   ├── skills/            # 内置策略（bull_trend / ma_golden_cross 等）
│   │   └── tools/             # Agent 工具集（分析 / 数据 / 市场 / 搜索）
│   ├── core/                  # 核心业务
│   │   ├── backtest/          # 回测引擎
│   │   ├── strategy_backtest/ # 策略回测（quantstats 可视化）
│   │   ├── advanced_backtest/ # 高级回测（多因子优化）
│   │   ├── pipeline.py        # 股票分析管线
│   │   ├── market_review.py   # 大盘复盘
│   │   └── market_strategy.py # 市场策略系统
│   ├── notification_sender/   # 通知推送（10+ 渠道）
│   ├── repositories/          # 数据访问层（SQLAlchemy）
│   └── services/              # 业务服务层
├── stock_selector/            # 选股系统
│   ├── strategies/
│   │   ├── Python/            # Python 策略（六维选股、龙头战法等 20+）
│   │   └── NLP/               # YAML 自然语言策略
│   ├── backtest_evaluator.py  # 选股回测评估器
│   └── main.py                # 选股入口
├── strategies/                # Agent 内置交易策略（YAML 配置）
├── tests/                     # 单元测试
└── watchdog/                  # 盘中实时监控
    ├── strategies/            # T+0 策略
    ├── monitor.py             # 行情监控器
    ├── risk_manager.py        # 风险管理器
    └── notifier.py            # 监控通知
```

## 📋 内置交易纪律

| 规则       | 说明                                                            |
| ---------- | --------------------------------------------------------------- |
| 严禁追高   | 乖离率超阈值（默认 5%，可配置）自动提示风险；强势趋势股自动放宽 |
| 趋势交易   | MA5 > MA10 > MA20 多头排列                                      |
| 精确点位   | 买入价、止损价、目标价                                          |
| 检查清单   | 每项条件以「满足 / 注意 / 不满足」标记                          |
| 新闻时效   | 可配置新闻最大时效（默认 3 天），避免使用过时信息               |

## 🚀 快速开始

### 环境要求

- **Python** 3.10+
- **Node.js** 18+（使用 WebUI 时需要）
- **Redis**（可选，用于缓存加速，默认开启）
- **wkhtmltopdf**（可选，用于 Markdown 报告转图片推送）

### 一、克隆仓库

```bash
git clone https://github.com/your-username/FinancialTool.git
cd FinancialTool
```

### 二、安装 Python 依赖

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

### 三、配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，至少需要配置：
#   1. LITELLM_MODEL + 至少一个 API Key（AI 模型）
#   2. STOCK_LIST（自选股列表）
#   3. 至少一个通知渠道（如 EMAIL_SENDER / EMAIL_PASSWORD）
```

### 四、安装前端依赖并构建（可选，使用 WebUI 时）

```bash
cd apps/dsa-web
npm install
npm run build
cd ../..
```

> 也可通过环境变量 `WEBUI_AUTO_BUILD=true` 让程序首次启动时自动构建。

### 五、运行

```bash
# 单次分析
python main.py

# 单次分析 + 启动 Web 服务
python main.py --serve

# 仅启动 Web 服务
python main.py --serve-only

# 定时任务模式（每日按时执行）
python main.py --schedule

# 仅运行大盘复盘
python main.py --market-review

# 指定分析特定股票
python main.py --stocks 600519,000001

# 调试模式
python main.py --debug

# 仅获取数据，不进行 AI 分析
python main.py --dry-run
```

### 六、Docker 部署

```bash
# 定时任务模式
docker-compose -f ./docker/docker-compose.yml up -d analyzer

# FastAPI Web 服务模式
docker-compose -f ./docker/docker-compose.yml up -d server

# 同时启动两种模式
docker-compose -f ./docker/docker-compose.yml up -d
```

## ⚙️ 配置说明

详细配置项请参考 `.env.example`，以下是核心配置类别的概述：

### AI 模型配置

通过 LiteLLM 统一调用，核心配置为 `LITELLM_MODEL`（主模型），可选 `LITELLM_FALLBACK_MODELS`（备选模型链）。

| 提供商     | 配置示例                                          |
| ---------- | ------------------------------------------------- |
| Gemini     | `LITELLM_MODEL=gemini/gemini-2.5-flash`           |
| Claude     | `LITELLM_MODEL=anthropic/claude-3-5-sonnet`       |
| OpenAI     | `LITELLM_MODEL=openai/gpt-4o`                     |
| DeepSeek   | `LITELLM_MODEL=openai/deepseek-chat`              |
| 通义千问   | `LITELLM_MODEL=openai/qwen-max`                   |
| AIHubMix   | `LITELLM_MODEL=openai/gemini-3.1-pro-preview`     |

支持多 Key 负载均衡（`*_API_KEYS`），以及 `*_BASE_URL` 自定义 API 地址。

### 数据源配置

按优先级顺序尝试多个数据源（数字越小优先级越高）：

| 数据源     | 默认优先级 | 说明                       |
| ---------- | :--------: | -------------------------- |
| efinance   |     0      | 东方财富（推荐首选）       |
| akshare    |     1      | 东方财富爬虫               |
| tushare    |     2      | Tushare Pro（需 Token）    |
| pytdx      |     2      | 通达信行情服务器            |
| baostock   |     3      | 证券宝                     |
| yfinance   |     4      | Yahoo Finance（美股推荐）  |

优先级可通过环境变量调整（如 `YFINANCE_PRIORITY=0`）。

### 通知渠道配置

支持以下推送渠道（可同时配置多个）：

| 渠道         | 配置方式                     | 说明                         |
| ------------ | ---------------------------- | ---------------------------- |
| 邮件         | `EMAIL_SENDER` + `PASSWORD`  | 支持 QQ / 163 / Gmail 等     |
| 企业微信     | `WECHAT_WEBHOOK_URL`         | 群机器人 Webhook             |
| 飞书         | `FEISHU_WEBHOOK_URL`         | 群机器人 Webhook             |
| Telegram     | `BOT_TOKEN` + `CHAT_ID`      | Bot API                      |
| 钉钉         | `CUSTOM_WEBHOOK_URLS`        | 群机器人 Webhook             |
| Discord      | `DISCORD_WEBHOOK_URL`        | 频道 Webhook                 |
| Pushover     | `USER_KEY` + `API_TOKEN`     | 需注册 Pushover 账号         |
| PushPlus     | `PUSHPLUS_TOKEN`             | 国内推送服务（推荐）         |
| Server酱3    | `SERVERCHAN3_SENDKEY`        | 微信推送                     |
| 自定义 Webhook | `CUSTOM_WEBHOOK_URLS`      | 任意 POST JSON 的 Webhook    |

支持 Markdown 转图片（需安装 wkhtmltopdf）、消息分批发送、股票分组邮件等功能。

### 定时任务配置

| 配置项                     | 说明                           | 默认值  |
| -------------------------- | ------------------------------ | :-----: |
| `SCHEDULE_ENABLED`         | 是否启用定时任务               | `false` |
| `SCHEDULE_TIME`            | 每日执行时间（HH:MM）           | `18:00` |
| `RUN_IMMEDIATELY`          | 启动时是否立即执行一次         | `true`  |
| `MARKET_REVIEW_ENABLED`    | 是否启用大盘复盘               | `true`  |
| `MARKET_REVIEW_REGION`     | 复盘市场区域                   |  `cn`   |
| `TRADING_DAY_CHECK_ENABLED`| 是否仅在交易日执行              | `false` |

### WebUI 配置

| 配置项              | 说明                     |    默认值    |
| ------------------- | ------------------------ | :----------: |
| `WEBUI_ENABLED`     | 是否默认启动 WebUI       |   `false`    |
| `WEBUI_HOST`        | 监听地址                 | `127.0.0.1`  |
| `WEBUI_PORT`        | 监听端口                 |    `8000`    |
| `WEBUI_AUTO_BUILD`  | 启动前自动构建前端       |    `true`    |
| `ADMIN_AUTH_ENABLED`| 是否启用登录密码保护     |   `false`    |

### 选股系统配置

- **策略类型**：支持 `PYTHON`（代码策略）和 `NATURAL_LANGUAGE`（YAML 策略），通过 `STOCK_SELECTOR_PREFERRED_STRATEGY_TYPE` 切换
- **六维选股权重**：各维度（主力交易 / 庄家控盘 / 动量 / 共振 / 强势爆发 / 板块）独立权重，范围 0.0 ~ 2.0
- **多线程加速**：`STOCK_SELECTOR_ENABLE_MULTITHREADING=true` + 可配置线程数
- **数据更新**：`STOCK_SELECTOR_UPDATE_DATA_DEFAULT_DAYS` 控制下载的历史交易日数

### 回测配置

| 配置项                       | 说明                 | 默认值 |
| ---------------------------- | -------------------- | :----: |
| `BACKTEST_ENABLED`           | 是否启用回测         | `true` |
| `BACKTEST_EVAL_WINDOW_DAYS`  | 评估窗口（交易日）    |  `10`  |
| `BACKTEST_MIN_AGE_DAYS`      | 仅回测 N 天前的记录   |  `14`  |
| `BACKTEST_NEUTRAL_BAND_PCT`  | 中性区间阈值（%）     | `2.0`  |

### 盘中监控配置（Watchdog）

盘中监控系统支持实时行情轮询和 T+0 策略：

| 配置项                         | 说明                   | 默认值 |
| ------------------------------ | ---------------------- | :----: |
| `INTRADAY_POLLING_INTERVAL`    | 分时行情轮询间隔（秒）   |  `5`   |
| `BATCH_DOWNLOAD_POLLING_INTERVAL` | 批量下载进度轮询间隔（秒） | `1` |
| `SCREEN_ASYNC_POLLING_INTERVAL`| 异步选股进度轮询间隔（秒） | `1` |

## 📱 支持的通知渠道

- Telegram（推荐）
- Discord
- 邮件（支持多接收人、股票分组）
- 企业微信
- 飞书
- 钉钉
- Pushover
- PushPlus（国内推荐）
- Server酱3（微信推送）
- 自定义 Webhook（支持 Bearer Token 认证）

## 🌐 Web 服务

启动 FastAPI 服务后：

- 前端页面：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs
- 功能页面：
  - **首页** — 股票分析、大盘复盘
  - **选股** — 智能选股系统
  - **分时** — 盘中实时行情
  - **回测** — 分析准确率评估
  - **可视化** — 回测图表展示
  - **对话** — Agent 策略问答
  - **历史** — 历史分析记录
  - **设置** — 系统配置管理

## 📊 选股系统

选股系统是本项目的一大特色模块，支持：

### 策略引擎
- **Python 策略**：20+ 内置策略，包括六维选股、龙头战法、均线金叉、放量突破、共振追涨等
- **YAML 自然语言策略**：用自然语言描述选股逻辑，适合非编程用户
- **策略回测**：对选股策略进行历史回测，输出胜率、盈亏比、最大回撤等指标

### 六维选股
六维选股从以下六个维度综合评分：
1. **主力交易** — 主力资金流入流出
2. **庄家控盘** — 筹码集中度分析
3. **动量** — 价格动量指标
4. **共振** — 多指标共振信号
5. **强势爆发** — 强势突破形态
6. **板块** — 板块效应分析

每个维度可独立配置权重，并设置最少匹配维度数。

### 数据管理
- 自动批量下载 A 股全市场历史数据
- 增量更新机制，避免重复下载
- 数据库索引优化，加速查询

## 🤖 Bot 命令

支持钉钉、飞书 Stream 模式的 Bot 交互：

| 命令                | 说明             |
| ------------------- | ---------------- |
| `/分析 股票代码`    | 分析指定股票     |
| `/批量 股票1,股票2` | 批量分析         |
| `/问股 问题`        | Agent 策略问答   |
| `/市场`             | 市场概览         |
| `/帮助`             | 查看所有命令     |
| `/状态`             | 查看系统运行状态 |

## 📄 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。

## ⚠️ 免责声明

本工具仅供**信息和教育目的使用**。分析结果由 AI 生成，不应被视为投资建议。股票市场投资有风险，您应该：

- 在做出投资决策前进行自己的研究
- 理解过去的表现并不保证未来的结果
- 只投资您能承受损失的资金
- 咨询持牌财务顾问获取个性化建议

本工具的开发者不对使用本软件造成的任何财务损失负责。

## 🙏 致谢

- 大量代码借鉴自 [daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis)
- [AkShare](https://github.com/akfamily/akshare) — 股票数据源
- [efinance](https://github.com/Micro-sheep/efinance) — 东方财富数据
- [Tushare](https://tushare.pro/) — 金融数据接口
- [LiteLLM](https://github.com/BerriAI/litellm) — 统一 LLM 调用
- [Tavily](https://tavily.com/) — 新闻搜索 API

---

**Made with ❤️ | 如果本项目对你有帮助，请给个 Star ⭐**