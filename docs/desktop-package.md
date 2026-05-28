# 桌面端打包说明 (Windows)

本项目可打包为 Windows 桌面应用，使用 Electron 作为桌面壳，`apps/dsa-web` 的 React UI 作为界面，Python FastAPI 作为后端服务。

## 架构说明

```
┌──────────────────────────────────────────────┐
│                  Electron                     │
│  ┌─────────────┐     ┌───────────────────┐  │
│  │ main.js     │────▶│ 启动后端进程       │  │
│  │ (主进程)    │     │ stock_analysis.exe │  │
│  └──────┬──────┘     └────────┬──────────┘  │
│         │                     │              │
│         ▼                     ▼              │
│  ┌─────────────┐     ┌───────────────────┐  │
│  │ BrowserWin  │◀────│ FastAPI (Python)  │  │
│  │ 加载 UI     │     │ 127.0.0.1:8000+   │  │
│  └─────────────┘     └───────────────────┘  │
└──────────────────────────────────────────────┘
```

- **React UI**：由 `apps/dsa-web` 使用 Vite 构建，输出到项目根目录的 `static/` 文件夹
- **后端服务**：`main.py` 通过 `--serve-only` 模式启动 FastAPI 服务，托管静态文件并提供 API
- **Electron 壳**：`apps/dsa-desktop/main.js` 启动时自动拉起后端进程，轮询 `/api/health` 直到就绪，然后加载 Web UI
- **便携模式**：用户配置 `.env` 和数据库放在 exe 同级目录，不依赖系统路径

### main.js 启动流程

1. 显示 `renderer/loading.html` 加载页
2. 在 8000-8100 范围内寻找可用端口
3. 检查 exe 同目录是否存在 `.env`，不存在则自动创建默认配置文件
4. 启动后端进程（打包模式用 `stock_analysis.exe`，开发模式用 `python main.py`）
5. 轮询 `http://127.0.0.1:{port}/api/health`，超时时间 60 秒
6. 后端就绪后加载 `http://127.0.0.1:{port}/`
7. 应用关闭时自动终止后端进程（Windows 使用 `taskkill /T /F`）

## 本地开发

### 一键启动（推荐）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-desktop.ps1
```

该脚本自动完成以下步骤：
1. 检查并安装前端依赖（`apps/dsa-web/node_modules`）
2. 构建 React UI 到 `static/`
3. 检查并安装 Electron 依赖（`apps/dsa-desktop/node_modules`）
4. 启动 Electron 开发模式（使用 `python main.py` 作为后端）

### 手动启动

**1) 构建 React UI：**

```powershell
cd apps\dsa-web
npm install
npm run build
```

构建产物输出到 `static/` 目录。

**2) 启动 Electron：**

```powershell
cd apps\dsa-desktop
npm install
npm run dev
```

Electron 启动后会自动以 `python main.py --serve-only --host 127.0.0.1 --port <port>` 启动后端。

### 开发模式下的环境变量

开发模式下，Electron 会设置以下环境变量传给后端：

| 环境变量 | 值 | 说明 |
|---|---|---|
| `DSA_DESKTOP_MODE` | `true` | 标记为桌面模式 |
| `ENV_FILE` | 用户数据目录下的 `.env` 路径 | 配置文件路径 |
| `DATABASE_PATH` | 用户数据目录下的数据库路径 | SQLite 数据库路径 |
| `LOG_DIR` | 用户数据目录下的 `logs/` | 日志目录 |
| `SCHEDULE_ENABLED` | `false` | 桌面模式禁用定时任务 |
| `WEBUI_ENABLED` | `false` | 桌面模式禁用旧版 WebUI |
| `BOT_ENABLED` | `false` | 桌面模式禁用 Bot |
| `DINGTALK_STREAM_ENABLED` | `false` | 桌面模式禁用钉钉 |
| `FEISHU_STREAM_ENABLED` | `false` | 桌面模式禁用飞书 |

## Windows 打包

### 前置条件

| 依赖 | 版本要求 | 说明 |
|---|---|---|
| Node.js | 18+ | 前端构建 + Electron 打包 |
| Python | 3.10+ | 后端运行 + PyInstaller 打包 |
| Windows 开发者模式 | — | electron-builder 需要创建符号链接 |

**开启开发者模式：** 设置 → 隐私和安全性 → 开发者选项 → 开发者模式

### 一键打包

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-all.ps1
```

该脚本按顺序执行以下两个步骤：

#### 步骤 1：构建后端 (`scripts\build-backend.ps1`)

1. 构建 React UI（`npm run build`，输出到 `static/`）
2. 安装 PyInstaller（如未安装）
3. 安装 Python 依赖（`pip install -r requirements.txt`）
4. 清理旧的 `dist\backend`、`dist\stock_analysis`、`build\stock_analysis`
5. 使用 PyInstaller 打包后端：

```powershell
python -m PyInstaller `
  --name stock_analysis `
  --onedir `
  --noconfirm `
  --noconsole `
  --add-data "static;static" `
  --collect-data litellm `
  --hidden-import=multipart `
  --hidden-import=multipart.multipart `
  --hidden-import=json_repair `
  --hidden-import=api `
  --hidden-import=api.app `
  --hidden-import=api.deps `
  --hidden-import=api.v1 `
  --hidden-import=api.v1.router `
  --hidden-import=api.v1.endpoints `
  --hidden-import=api.v1.endpoints.analysis `
  --hidden-import=api.v1.endpoints.history `
  --hidden-import=api.v1.endpoints.stocks `
  --hidden-import=api.v1.endpoints.health `
  --hidden-import=api.v1.schemas `
  --hidden-import=api.v1.schemas.analysis `
  --hidden-import=api.v1.schemas.history `
  --hidden-import=api.v1.schemas.stocks `
  --hidden-import=api.v1.schemas.common `
  --hidden-import=api.middlewares `
  --hidden-import=api.middlewares.error_handler `
  --hidden-import=src.services `
  --hidden-import=src.services.task_queue `
  --hidden-import=src.services.analysis_service `
  --hidden-import=src.services.history_service `
  --hidden-import=uvicorn.logging `
  --hidden-import=uvicorn.loops `
  --hidden-import=uvicorn.loops.auto `
  --hidden-import=uvicorn.protocols `
  --hidden-import=uvicorn.protocols.http `
  --hidden-import=uvicorn.protocols.http.auto `
  --hidden-import=uvicorn.protocols.websockets `
  --hidden-import=uvicorn.protocols.websockets.auto `
  --hidden-import=uvicorn.lifespan `
  --hidden-import=uvicorn.lifespan.on `
  main.py
```

> **注意：** PyInstaller 使用 `--onedir` 模式（非 `--onefile`），产物为 `dist/stock_analysis/` 目录，其中包含 `stock_analysis.exe` 及所有依赖。

6. 将 `dist/stock_analysis/` 整体复制到 `dist/backend/stock_analysis/`

#### 步骤 2：构建桌面应用 (`scripts\build-desktop.ps1`)

1. 检查 Windows 开发者模式是否开启（可设置 `DSA_SKIP_DEVMODE_CHECK=true` 跳过检查）
2. 设置 electron-builder 缓存目录为 `.electron-builder-cache`
3. 验证 `dist/backend/stock_analysis` 是否存在
4. 安装 Electron 依赖
5. 停止正在运行的旧进程（`Daily Stock Analysis` 和 `stock_analysis`）
6. 清理旧的 `dist/win-unpacked`
7. 确保 `app-builder.exe` 存在（不存在则重装依赖）
8. 执行 `npx electron-builder --win nsis`

产物位于 `apps/dsa-desktop/dist/`：
- `win-unpacked/` — 免安装版（绿色版）
- `Daily Stock Analysis Setup x.x.x.exe` — NSIS 安装包

### electron-builder 配置说明

配置定义在 [package.json](file:///e:/工作/Code/FinancialTool/apps/dsa-desktop/package.json) 的 `build` 字段：

```json
{
  "build": {
    "appId": "com.daily-stock-analysis.desktop",
    "productName": "Daily Stock Analysis",
    "directories": { "output": "dist" },
    "files": ["main.js", "preload.js", "renderer/**/*"],
    "extraResources": [
      { "from": "../../.env.example", "to": ".env.example" },
      { "from": "../../dist/backend/stock_analysis", "to": "backend/stock_analysis" }
    ],
    "win": { "target": "nsis" },
    "mac": { "target": "dmg" }
  }
}
```

- `extraResources` 会将 `.env.example`（如存在）和后端 `stock_analysis/` 目录打包进 `resources/`
- Windows 平台目标为 `nsis`（生成安装包和免安装版）

> **注意：** `.env.example` 文件是**可选的**。如果项目根目录不存在该文件，打包时该条目会被静默忽略。运行时如果 `resources/.env.example` 也不存在，Electron 会自动创建一个包含基本注释的默认 `.env` 文件。

### 分步手动打包

如果需要在 CI 或其他场景下分步执行：

**1) 构建 React UI：**

```powershell
cd apps\dsa-web
npm install
npm run build
```

**2) 打包 Python 后端：**

```powershell
# 安装依赖
pip install pyinstaller
pip install -r requirements.txt

# PyInstaller 打包（--onedir 模式）
python -m PyInstaller --name stock_analysis --onedir --noconfirm --noconsole `
  --add-data "static;static" --collect-data litellm `
  --hidden-import=multipart --hidden-import=multipart.multipart `
  --hidden-import=json_repair `
  --hidden-import=api --hidden-import=api.app --hidden-import=api.deps `
  --hidden-import=api.v1 --hidden-import=api.v1.router `
  --hidden-import=api.v1.endpoints --hidden-import=api.v1.endpoints.analysis `
  --hidden-import=api.v1.endpoints.history --hidden-import=api.v1.endpoints.stocks `
  --hidden-import=api.v1.endpoints.health `
  --hidden-import=api.v1.schemas --hidden-import=api.v1.schemas.analysis `
  --hidden-import=api.v1.schemas.history --hidden-import=api.v1.schemas.stocks `
  --hidden-import=api.v1.schemas.common `
  --hidden-import=api.middlewares --hidden-import=api.middlewares.error_handler `
  --hidden-import=src.services --hidden-import=src.services.task_queue `
  --hidden-import=src.services.analysis_service --hidden-import=src.services.history_service `
  --hidden-import=uvicorn.logging --hidden-import=uvicorn.loops --hidden-import=uvicorn.loops.auto `
  --hidden-import=uvicorn.protocols --hidden-import=uvicorn.protocols.http --hidden-import=uvicorn.protocols.http.auto `
  --hidden-import=uvicorn.protocols.websockets --hidden-import=uvicorn.protocols.websockets.auto `
  --hidden-import=uvicorn.lifespan --hidden-import=uvicorn.lifespan.on `
  main.py

# 复制产物到 backend 目录
mkdir dist\backend
xcopy /E /I dist\stock_analysis dist\backend\stock_analysis
```

**3) 打包 Electron 桌面应用：**

```powershell
cd apps\dsa-desktop
npm install
npm run build
```

## 打包产物目录结构

打包后用户拿到的完整目录结构（`win-unpacked/`）：

```
win-unpacked/
├── Daily Stock Analysis.exe    ← 双击启动
├── .env                        ← 用户配置文件（首次启动自动生成）
├── data/
│   └── stock_analysis.db       ← SQLite 数据库
├── logs/
│   └── desktop.log             ← 运行日志
└── resources/
    └── backend/
        └── stock_analysis/     ← 后端目录（PyInstaller --onedir 产物）
            ├── stock_analysis.exe
            ├── _internal/
            ├── python3xx.dll
            └── ... (其他依赖)
```

## 配置文件说明

桌面版启动时会自动处理配置文件：

1. 检查 exe 同级目录是否存在 `.env`
2. 如果不存在，尝试从 `resources/.env.example` 复制
3. 如果 `.env.example` 也不存在，自动创建包含基本注释的默认 `.env`

用户需要编辑 `.env` 配置以下关键项：

| 配置项 | 说明 | 是否必需 |
|---|---|---|
| `TUSHARE_TOKEN` | 数据源 Token（[tushare.pro](https://tushare.pro) 注册获取） | 推荐（缺少则降级到其他数据源） |
| `GEMINI_API_KEY` / `LITELLM_MODEL` | AI 分析 API Key | AI 诊股/分析功能必需 |
| `STOCK_LIST` | 自选股列表（逗号分隔，如 `600519,000001`） | 选股范围 |
| `LOG_LEVEL` | 日志级别（默认 `INFO`） | 否 |

> **注意：** 即使不配置 API Key，桌面版的核心功能（行情查看、分时图、技术指标、策略筛选）仍可使用。仅 AI 诊股、智能分析等功能需要配置 LLM 的 API Key。

## 常见问题

### 启动后一直显示 "Preparing backend..." 加载页

1. 检查 `logs/desktop.log` 查看详细错误信息
2. 确认 `.env` 文件存在且配置了有效的 API Key
3. 确认端口 8000-8100 未被其他程序占用
4. 尝试在终端手动运行 `resources/backend/stock_analysis/stock_analysis.exe --serve-only --host 127.0.0.1 --port 8000` 查看输出

### 后端启动报 ModuleNotFoundError

PyInstaller 打包时可能遗漏了某些动态导入的模块。解决方法：

1. 找到缺失的模块名
2. 在 [scripts/build-backend.ps1](file:///e:/工作/Code/FinancialTool/scripts/build-backend.ps1) 的 `$hiddenImports` 数组中添加对应的 `--hidden-import`
3. 重新执行打包

### 打包时提示 Developer Mode 未开启

```powershell
# 跳过开发者模式检查（仅在确认不会产生符号链接问题时使用）
$env:DSA_SKIP_DEVMODE_CHECK = 'true'
powershell -ExecutionPolicy Bypass -File scripts\build-all.ps1
```

或在 Windows 设置中开启：设置 → 隐私和安全性 → 开发者选项 → 开发者模式。

### electron-builder 报 app-builder.exe 缺失

构建脚本已自动处理：如果检测到 `app-builder.exe` 缺失，会自动重装依赖。如果仍有问题：

```powershell
cd apps\dsa-desktop
Remove-Item -Recurse -Force node_modules
npm install
```

### UI 加载空白

1. 确认 `static/index.html` 存在（如不存在需重新构建 React UI）
2. 确认后端启动成功：访问 `http://127.0.0.1:8000/` 检查
3. 检查 `logs/desktop.log` 确认后端是否正常监听端口

### 打包后后端 exe 体积过大

PyInstaller `--onedir` 模式下打包了整个 Python 环境。如果想减小体积，可以：

- 使用 `--exclude-module` 排除不需要的模块
- 在虚拟环境中只安装必需的依赖后再打包

### 打包产物中缺少 .env.example

该文件是可选的。如果不存在，用户首次启动时会自动生成默认配置文件。如需在打包时包含模板，在项目根目录创建 `.env.example` 文件即可。

## 分发给用户

打包产物的分发方式有两种：安装包（推荐）和绿色版。

### 安装包（推荐）

产物为 `apps/dsa-desktop/dist/Daily Stock Analysis Setup x.x.x.exe`。

用户使用步骤：

1. **双击安装程序**，按提示完成安装（默认路径 `C:\Program Files\Daily Stock Analysis\`）
2. **首次启动**桌面应用
3. **配置 API Key**：打开安装目录，编辑 `Daily Stock Analysis.exe` 同级目录下的 `.env` 文件
4. **重新启动**应用即可正常使用

### 绿色版（免安装）

将 `apps/dsa-desktop/dist/win-unpacked/` 整个文件夹打包为 zip 发给用户。

用户使用步骤：

1. 解压文件夹到任意位置
2. 双击 `Daily Stock Analysis.exe` 启动
3. 首次启动后关闭应用，编辑同目录下的 `.env` 配置 API Key 和股票列表
4. 重新启动应用即可正常使用