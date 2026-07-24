# Agentic RAG Assistant

一个使用 OpenAI 兼容 Chat Completions API、本地 FAISS 知识库和工具调用构建的中文 AI Agent。它能在对话中自行选择安全计算器或本地知识库检索，并保留同一会话的上下文。

## 功能

- 多轮对话与按 `session_id` 隔离的会话记忆
- 模型驱动的工具调用，可在一轮中组合多个工具
- 安全计算器，支持 `+`、`-`、`*`、`/`、`//`、`%`、`**` 与正负号
- 本地 RAG：读取 PDF、Markdown、TXT，切分后写入 FAISS 索引
- 检索结果携带来源、PDF 页码、FAISS 距离和参考相似度
- 原生 HTTP 网页、Gradio 管理界面与 Gradio 检索调试页
- Agent 端到端评测和 RAG 检索评测，生成 JSON、Markdown 或 CSV 报告

## 项目结构

```text
.
├── agent.py                  # Agent 循环、模型调用和工具调度
├── client.py                 # OpenAI 兼容客户端
├── memory.py                 # 会话记忆
├── prompt.py                 # 系统提示词
├── tools.py                  # 计算器与知识库工具注册
├── tools_config.py           # OpenAI Tool Calling Schema
├── tool_executor.py          # 工具参数解析、执行与调用记录
├── build_index.py            # 读取文档并构建 FAISS 索引
├── knowledge_base.py         # FAISS 加载、阈值过滤与检索格式化
├── embedding.py              # EmbeddingModel 轻量封装
├── main.py                   # 命令行对话入口
├── web_server.py             # 原生 HTTP API 与 web.html 服务
├── web.html                  # 原生 HTML/CSS/JavaScript 网页界面
├── rag_debug.py              # Gradio 检索调试界面
├── evaluate_retrieval.py     # RAG 检索评测
├── evaluation/
│   ├── run_evaluation.py     # Agent 端到端评测
│   ├── test_cases.json
│   └── results/latest_report.md
├── data/
│   ├── knowledge/            # 待索引的 PDF、Markdown、TXT
│   └── evaluation/rag_cases.json
├── faiss_index/              # 构建后的 index.faiss 与 index.pkl
├── config.example.toml
└── requirements.txt
```

## 环境安装

需要 Python 3.11 或更高版本；项目当前在 Python 3.13 环境中验证。

```bash
cd "ai agent"
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`sentence-transformers` 会安装 PyTorch 等运行依赖。若需要 GPU 推理，请按目标平台从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应版本后，再安装本项目依赖。

## 配置

在项目根目录创建 `config.toml`。它包含 API 凭证，因此已被 `.gitignore` 忽略；不要提交真实密钥。

```toml
model_provider = "OpenAI"
model = "your-chat-model"

[model_providers.OpenAI]
api_key = "your-api-key"
base_url = "https://your-openai-compatible-endpoint/v1"
model = "your-chat-model"
timeout = 30.0
max_retries = 1

[embedding]
# 该目录必须已经存在，并包含本地 Hugging Face Embedding 模型文件。
model_path = "/absolute/path/to/bge-small-zh-v1.5"
data_path = "data/knowledge"
index_path = "faiss_index"
top_k = 3
score_threshold = 1.0
chunk_size = 600
chunk_overlap = 100

[ui]
host = "0.0.0.0"
port = 7860
share = false
```

模型服务必须支持 OpenAI Chat Completions 的工具调用格式。`model_path` 是本地 Embedding 模型目录；建库端和检索端必须使用同一模型。相对路径均以项目根目录为基准。

`config.example.toml` 可作为本地配置文件的起点，但运行本项目时应使用上面的顶层 `model_provider` 和 `model` 字段，以及 `[model_providers.OpenAI]` 配置段。

## 构建知识库

将 `.pdf`、`.md` 或 `.txt` 文件放到 `data/knowledge/`（支持子目录），然后执行：

```bash
python build_index.py
```

脚本会加载文档、按配置切分文本、生成归一化向量，并在 `faiss_index/` 生成 `index.faiss` 和 `index.pkl`。重建会删除该目录中的旧索引。索引仅应从可信来源加载，因为 FAISS 反序列化会读取 `index.pkl`。

## 启动

所有命令都应在项目根目录执行，并先完成配置和索引构建。

### 命令行

```bash
python main.py
```

输入 `e` 或 `q` 退出。命令行使用固定的 `cli` 会话，因此连续输入会保留上下文。

### 原生网页

```bash
python web_server.py
```

默认监听 `0.0.0.0:8000`；若端口被占用，会依次尝试后续 9 个端口。可指定地址与起始端口：

```bash
python web_server.py --host 127.0.0.1 --port 8080
```

打开终端输出的地址即可使用 `web.html` 界面。该界面提供对话、新建会话、文档上传、删除和重建索引；单次上传总大小不超过 20 MB。

### Gradio 网页

```bash
python web.py
```

监听地址、端口和公开分享开关由 `[ui]` 中的 `host`、`port`、`share` 控制，默认端口为 `7860`。界面支持聊天、快捷提问、上传资料、查看/删除资料及重建知识库。

### 检索调试页

```bash
python rag_debug.py
```

默认在 `http://127.0.0.1:7861` 启动。它展示 Top-K 检索结果、距离、来源、页码、阈值判定和文本内容，方便调整 `top_k` 与 `score_threshold`。

## 评测

### Agent 端到端评测

```bash
python evaluation/run_evaluation.py
```

测试用例来自 `evaluation/test_cases.json`。评测检查工具调用、最终回答关键词、原始检索关键词、模型错误和人工复核标记，并将带时间戳的 JSON 报告及 `latest_report.md` 写入 `evaluation/results/`。

### RAG 检索评测

```bash
python evaluate_retrieval.py
```

测试用例来自 `data/evaluation/rag_cases.json`。脚本检查来源命中、关键词覆盖、阈值判断、命中排名和 MRR，并将 `rag_eval_v2_summary.csv` 与 `rag_eval_v2_details.csv` 写入 `reports/`。

## 工具与检索行为

| 工具 | 用途 |
| --- | --- |
| `calculator` | 用 AST 白名单解析数学表达式，拒绝函数调用和任意代码执行。 |
| `search_knowledge` | 以配置的 Top-K 查询 FAISS，并过滤距离大于 `score_threshold` 的文本块。 |

每次 Agent 执行最多进行 8 轮模型/工具交互。若工具已经成功执行、但模型在整理最终回答时失败，Agent 会返回已取得的工具结果作为降级回答。

## 常见问题

- **找不到 `config.toml`**：按“配置”章节创建文件，并确认它位于项目根目录。
- **找不到 Embedding 模型或 FAISS 索引**：检查 `[embedding]` 路径，先准备本地模型，再运行 `python build_index.py`。
- **检索不到相关内容**：确认资料已被重新建库；随后适当增大 `top_k` 或放宽 `score_threshold`。FAISS 距离越小，结果通常越相关。
- **模型调用失败**：检查 API Key、`base_url`、模型名称，以及服务端是否支持工具调用。
