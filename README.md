<div align="center">

# 金融工具 - AI 股票智能分析系统

**AI-powered stock analysis system for A-shares / Hong Kong / US stocks**

基于 AI 大模型的 A股/港股/美股自选股智能分析系统，每日自动分析并推送「决策仪表盘」到多渠道。

[功能特性](#-功能特性) · [快速开始](#-快速开始) · [技术栈](#-技术栈与数据来源) · [免责声明](#-免责声明)

</div>

## 📖 项目说明

本项目从 [https://github.com/ZhuLinsen/daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis) 处借鉴了大量代码，在此基础上进行开发和优化。

## ✨ 功能特性

| 模块       | 功能         | 说明                                                           |
| ---------- | ------------ | -------------------------------------------------------------- |
| AI         | 决策仪表盘   | 一句话核心结论 + 精确买卖点位 + 操作检查清单                   |
| 分析       | 多维度分析   | 技术面（盘中实时 MA/多头排列）+ 筹码分布 + 舆情情报 + 实时行情 |
| 市场       | 全球市场     | 支持 A股、港股、美股及美股指数（SPX、DJI、IXIC 等）            |
| 策略       | 市场策略系统 | 内置 A股「三段式复盘策略」与美股「Regime Strategy」            |
| 复盘       | 大盘复盘     | 每日市场概览、板块涨跌；支持 cn(A股)/us(美股)/both(两者) 切换  |
| 智能导入   | 多源导入     | 支持图片、CSV/Excel 文件、剪贴板粘贴；Vision LLM 提取代码+名称 |
| 回测       | AI 回测验证  | 自动评估历史分析准确率，方向胜率、止盈止损命中率               |
| Agent 问股 | 策略对话     | 多轮策略问答，支持均线金叉/缠论/波浪等 11 种内置策略           |
| 推送       | 多渠道通知   | 企业微信、飞书、Telegram、钉钉、邮件、Pushover、Discord 等     |
| 自动化     | 定时运行     | GitHub Actions 定时执行，无需服务器                            |
| WebUI      | 管理界面     | React + FastAPI 完整 Web 应用                                  |
| 桌面应用   | 跨平台桌面端 | Electron 桌面应用，Windows/macOS 支持                          |

## 🔧 技术栈与数据来源

| 类型     | 支持                                                                                  |
| -------- | ------------------------------------------------------------------------------------- |
| AI 模型  | AIHubMix、Gemini、OpenAI 兼容、DeepSeek、通义千问、Claude 等（统一通过 LiteLLM 调用） |
| 行情数据 | AkShare、Tushare、Pytdx、Baostock、YFinance                                           |
| 新闻搜索 | Tavily、SerpAPI、Bocha、Brave、MiniMax                                                |
| 后端框架 | FastAPI                                                                               |
| 前端框架 | React + TypeScript + Tailwind CSS                                                     |
| 桌面应用 | Electron                                                                              |

## 📋 内置交易纪律

| 规则     | 说明                                                            |
| -------- | --------------------------------------------------------------- |
| 严禁追高 | 乖离率超阈值（默认 5%，可配置）自动提示风险；强势趋势股自动放宽 |
| 趋势交易 | MA5 > MA10 > MA20 多头排列                                      |
| 精确点位 | 买入价、止损价、目标价                                          |
| 检查清单 | 每项条件以「满足 / 注意 / 不满足」标记                          |
| 新闻时效 | 可配置新闻最大时效（默认 3 天），避免使用过时信息               |

## 🚀 快速开始

### 方式一：GitHub Actions（推荐，零成本）

**无需服务器，每天自动运行！**

#### 1. Fork 本仓库

点击右上角 Fork 按钮

#### 2. 配置 Secrets

Settings → Secrets and variables → Actions → New repository secret

**AI 模型配置（至少配置一个）**

| Secret 名称           | 说明                                             | 必填 |
| --------------------- | ------------------------------------------------ | :--: |
| `AIHUBMIX_KEY`      | AIHubMix API Key，一 Key 切换使用全系模型        | 可选 |
| `GEMINI_API_KEY`    | Google AI Studio 获取免费 Key                    | 可选 |
| `ANTHROPIC_API_KEY` | Anthropic Claude API Key                         | 可选 |
| `OPENAI_API_KEY`    | OpenAI 兼容 API Key（支持 DeepSeek、通义千问等） | 可选 |

**通知渠道配置（至少配置一个）**

| Secret 名称             | 说明                 | 必填 |
| ----------------------- | -------------------- | :--: |
| `WECHAT_WEBHOOK_URL`  | 企业微信 Webhook URL | 可选 |
| `FEISHU_WEBHOOK_URL`  | 飞书 Webhook URL     | 可选 |
| `TELEGRAM_BOT_TOKEN`  | Telegram Bot Token   | 可选 |
| `TELEGRAM_CHAT_ID`    | Telegram Chat ID     | 可选 |
| `EMAIL_SENDER`        | 发件人邮箱           | 可选 |
| `EMAIL_PASSWORD`      | 邮箱授权码           | 可选 |
| `DISCORD_WEBHOOK_URL` | Discord Webhook URL  | 可选 |

**股票列表配置**

| Secret 名称    | 说明                                     | 必填 |
| -------------- | ---------------------------------------- | :--: |
| `STOCK_LIST` | 自选股代码，例如 `600519,AAPL,hk00700` |  ✅  |

#### 3. 启用 Actions

Actions 标签页 → 点击 `I understand my workflows, go ahead and enable them`

#### 4. 手动测试

Actions → `Daily Stock Analysis` → `Run workflow` → 选择模式 → `Run workflow`

### 方式二：本地部署

#### 1. 克隆仓库

```bash
git clone https://github.com/your-username/FinancialTool.git
cd FinancialTool
```

#### 2. 安装依赖

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

#### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件配置所需参数
```

#### 4. 运行

```bash
# 单次分析
python main.py

# 定时任务模式
python main.py --schedule

# 仅启动 Web 服务
python main.py --serve-only
```

## 📱 支持的通知渠道

- Telegram（推荐）
- Discord
- 邮件
- 企业微信
- 飞书
- 钉钉
- Pushover
- PushPlus
- ServerChan
- 自定义 Webhook

## 🌐 Web 服务

启动 FastAPI 服务：

```bash
python main.py --serve
```

- 访问地址：http://127.0.0.1:8000
- API 文档：http://127.0.0.1:8000/docs

## 🤖 Bot 命令

支持钉钉、飞书 Stream 模式的 Bot 交互：

- `/分析 股票代码` - 分析指定股票
- `/批量 股票1,股票2` - 批量分析
- `/问股 问题` - Agent 策略问答
- `/市场` - 市场概览

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## ⚠️ 免责声明

本工具仅供**信息和教育目的使用**。分析结果由 AI 生成，不应被视为投资建议。股票市场投资有风险，您应该：

- 在做出投资决策前进行自己的研究
- 理解过去的表现并不保证未来的结果
- 只投资您能承受损失的资金
- 咨询持牌财务顾问获取个性化建议

本工具的开发者不对使用本软件造成的任何财务损失负责。

## 🙏 致谢

- 大量代码借鉴自 [daily_stock_analysis](https://github.com/ZhuLinsen/daily_stock_analysis)
- [AkShare](https://github.com/akfamily/akshare) - 股票数据源
- [Google Gemini](https://ai.google.dev/) - AI 分析引擎
- [Tavily](https://tavily.com/) - 新闻搜索 API

---

**Made with ❤️ | 如果本项目对你有帮助，请给个 Star ⭐**
