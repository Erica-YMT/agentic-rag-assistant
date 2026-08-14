# Agentic RAG Assistant 部署说明

本文档记录 Agentic RAG Assistant 使用 Docker Compose 进行本地部署、日常启动、代码更新和故障排查的方法。

---

## 1. 部署架构

Docker Compose 启动两个服务：

```text
api
└── FastAPI
    └── 宿主机端口 8000

web
└── Web Server
    └── 宿主机端口 8001
```

容器内部调用关系：

```text
浏览器
   ↓
web 容器
   ↓
http://api:8000
   ↓
api 容器
   ↓
Agent / 大语言模型 / FAISS
```

外部访问地址：

```text
Web 页面：http://127.0.0.1:8001
API 文档：http://127.0.0.1:8000/docs
健康检查：http://127.0.0.1:8000/health
```

---

## 2. 环境要求

需要准备：

- Windows 10 或 Windows 11；
- WSL2；
- Ubuntu；
- Docker Desktop；
- Docker Desktop 的 WSL Integration；
- 本地 `bge-small-zh-v1.5` Embedding 模型；
- 可用的 OpenAI 兼容 Chat Completions API；
- 已填写的本地配置和 Docker 配置。

测试 Docker 是否可用：

```bash
docker run --rm hello-world
```

能够看到 Docker 欢迎信息，说明 WSL 已成功连接 Docker Desktop。

---

## 3. 项目目录

进入项目根目录：

```bash
cd "$HOME/ai/ai agent"
```

检查部署相关文件：

```bash
ls -lh \
  Dockerfile \
  docker-compose.yml \
  docker-up.sh \
  dc.sh \
  requirements-docker.txt \
  config.toml \
  config.docker.toml
```

主要部署文件：

| 文件 | 作用 |
|---|---|
| `Dockerfile` | 定义 Python 运行镜像和依赖安装过程 |
| `docker-compose.yml` | 编排 API 和 Web 两个容器 |
| `scripts/docker-up.sh` | 首次检查、构建并启动项目 |
| `scripts/dc.sh` | 日常执行 Docker Compose 命令 |
| `requirements-docker.txt` | Docker 运行环境使用的 Python 依赖 |
| `config.docker.toml` | 容器内部使用的真实配置 |
| `config.docker.example.toml` | Docker 配置模板 |

---

## 4. 配置文件

项目包含两组配置文件。

### 本地运行配置

```text
config.example.toml
config.toml
```

其中：

- `config.example.toml` 是可以提交到 GitHub 的配置模板；
- `config.toml` 是本地真实配置；
- `config.toml` 包含真实 API Key，不能提交。

### Docker 运行配置

```text
config.docker.example.toml
config.docker.toml
```

其中：

- `config.docker.example.toml` 是 Docker 配置模板；
- `config.docker.toml` 是 Docker 容器实际读取的配置；
- `config.docker.toml` 包含真实 API Key，不能提交。

Docker 配置中的关键路径：

```toml
[embedding]
model_path = "/models/bge-small-zh-v1.5"
data_path = "/app/data/knowledge"
index_path = "/app/faiss_index"

[api_client]
base_url = "http://api:8000"
```

注意：

- `embedding.model_path` 是容器内部模型路径；
- `embedding.data_path` 是容器内部知识文档路径；
- `embedding.index_path` 是容器内部 FAISS 索引路径；
- `[api_client].base_url` 使用 Compose 服务名 `api`；
- 模型提供商的 `base_url` 仍然填写真实的大语言模型接口地址。

确认真实配置已被 Git 忽略：

```bash
git check-ignore -v \
  config.toml \
  config.docker.toml
```

---

## 5. Docker 挂载关系

Docker Compose 使用目录挂载保存数据。

```text
宿主机 ./data
    ↓
容器 /app/data
```

```text
宿主机 ./faiss_index
    ↓
容器 /app/faiss_index
```

```text
宿主机 Embedding 模型目录
    ↓
容器 /models/bge-small-zh-v1.5
```

因此，以下数据不会因为容器被删除而消失：

```text
data/chat_history.db
data/user_memory.db
data/knowledge/
faiss_index/
```

---

## 6. 首次构建和启动

进入项目目录：

```bash
cd "$HOME/ai/ai agent"
```

执行：

```bash
./scripts/docker-up.sh
```

脚本会自动：

1. 检查部署所需文件；
2. 读取 `config.toml` 中的本地 Embedding 模型路径；
3. 设置 `EMBEDDING_MODEL_PATH`；
4. 创建 `data/knowledge/`；
5. 创建 `faiss_index/`；
6. 检查 Docker Compose 配置；
7. 构建 API 和 Web 镜像；
8. 启动两个容器；
9. 显示容器状态和访问地址。

首次构建需要安装：

- PyTorch；
- Sentence Transformers；
- SciPy；
- scikit-learn；
- LangChain；
- FAISS；
- FastAPI；
- Uvicorn。

首次构建的下载量较大，网络较慢时可能持续较长时间。

---

## 7. 日常使用

每次打开新的 WSL 终端后，先进入项目：

```bash
cd "$HOME/ai/ai agent"
```

### 启动项目

```bash
./scripts/dc.sh up -d
```

含义：

- `up`：创建或启动容器；
- `-d`：在后台运行。

### 查看容器状态

```bash
./scripts/dc.sh ps
```

正常状态类似：

```text
agentic-rag-api   Up ... healthy
agentic-rag-web   Up ...
```

状态说明：

| 状态 | 含义 |
|---|---|
| `Up` | 容器正在运行 |
| `healthy` | API 健康检查通过 |
| `Exited` | 容器已经停止 |
| `Restarting` | 容器正在反复崩溃重启 |

### 打开项目

```text
Web 页面：http://127.0.0.1:8001
API 文档：http://127.0.0.1:8000/docs
```

### 日常停止

```bash
./scripts/dc.sh stop
```

`stop` 只停止容器，不删除容器。

下次继续运行：

```bash
./scripts/dc.sh up -d
```

### 完整关闭

```bash
./scripts/dc.sh down
```

`down` 会：

- 停止容器；
- 删除容器；
- 删除 Compose 网络。

但不会删除：

- Docker 镜像；
- 项目代码；
- 聊天历史；
- 长期记忆；
- 知识文档；
- FAISS 索引；
- 本地 Embedding 模型。

---

## 8. 查看日志

### 同时查看 API 和 Web 实时日志

```bash
./scripts/dc.sh logs -f api web
```

按：

```text
Ctrl + C
```

只会退出日志查看，不会停止容器。

### 查看 API 最近 100 行日志

```bash
./scripts/dc.sh logs --tail=100 api
```

### 查看 Web 最近 100 行日志

```bash
./scripts/dc.sh logs --tail=100 web
```

### 查看全部服务最近 200 行

```bash
./scripts/dc.sh logs --tail=200
```

---

## 9. 为什么使用 `scripts/dc.sh`

`docker-compose.yml` 需要读取：

```text
EMBEDDING_MODEL_PATH
```

直接执行：

```bash
docker compose ps
```

可能出现：

```text
required variable EMBEDDING_MODEL_PATH is missing a value
```

`scripts/dc.sh` 会自动：

1. 从 `config.toml` 读取宿主机 Embedding 模型路径；
2. 检查模型目录是否存在；
3. 找不到时尝试从 ModelScope 缓存中搜索；
4. 设置 `EMBEDDING_MODEL_PATH`；
5. 再执行 Docker Compose 命令。

因此日常统一使用：

```bash
./scripts/dc.sh ...
```

例如：

```bash
./scripts/dc.sh up -d
./scripts/dc.sh ps
./scripts/dc.sh logs --tail=100 api
./scripts/dc.sh stop
./scripts/dc.sh down
```

---

## 10. 健康检查

检查 API：

```bash
curl -sS http://127.0.0.1:8000/health \
  | python -m json.tool --no-ensure-ascii
```

检查 Web：

```bash
curl -I http://127.0.0.1:8001
```

查看容器状态：

```bash
./scripts/dc.sh ps
```

API 正常时应显示：

```text
Up ... healthy
```

---

## 11. 功能验收

### 普通聊天

```bash
curl -sS \
  -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "deployment-test",
    "question": "你好，请介绍一下这个项目。"
  }' | python -m json.tool --no-ensure-ascii
```

### RAG 检索

```bash
curl -sS \
  -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "deployment-rag-test",
    "question": "项目主要包含哪些模块？"
  }' | python -m json.tool --no-ensure-ascii
```

返回结果中可以检查：

```text
called_tools
tool_calls
search_knowledge
```

### 直接检索知识库

```bash
curl -sS \
  -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "build_index.py 的作用是什么？",
    "top_k": 3
  }' | python -m json.tool --no-ensure-ascii
```

### 查看历史会话

```bash
curl -sS http://127.0.0.1:8000/history/sessions \
  | python -m json.tool --no-ensure-ascii
```

### 搜索聊天记录

```bash
curl -sS \
  "http://127.0.0.1:8000/history/search?q=FAISS" \
  | python -m json.tool --no-ensure-ascii
```

### 测试显式长期记忆

```bash
curl -sS \
  -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "deployment-memory-test",
    "question": "请记住：这个项目使用 Docker Compose 部署。"
  }' | python -m json.tool --no-ensure-ascii
```

查看长期记忆：

```bash
curl -sS http://127.0.0.1:8000/memory \
  | python -m json.tool --no-ensure-ascii
```

---

## 12. 修改代码后重新部署

Dockerfile 使用：

```dockerfile
COPY . ./
```

本地代码修改后，容器不会自动读取新代码，需要重新构建镜像。

修改以下文件后：

```text
app/agent/agent.py
api.py
app/memory/chat_memory.py
rag/knowledge_base.py
web_server.py
web.html
explicit_app/memory/chat_memory.py
user_app/memory/chat_memory.py
```

执行：

```bash
./scripts/dc.sh up --build -d
```

完成后检查：

```bash
./scripts/dc.sh ps
```

查看日志：

```bash
./scripts/dc.sh logs --tail=100 api web
```

---

## 13. 修改依赖后重新部署

修改以下文件后：

```text
Dockerfile
requirements-docker.txt
```

执行：

```bash
./scripts/dc.sh build --no-cache api web
./scripts/dc.sh up -d
```

`--no-cache` 会完全重新执行依赖安装，耗时可能较长。

普通 Python 或 HTML 代码修改不需要使用 `--no-cache`。

---

## 14. 本地知识文档发生变化

将新文件放入：

```text
data/knowledge/
```

支持：

- PDF；
- Markdown；
- TXT。

可以在本地重建索引：

```bash
python build_index.py
```

也可以在容器中执行：

```bash
./scripts/dc.sh exec api python build_index.py
```

完成后可以重启 API：

```bash
./scripts/dc.sh restart api
```

项目也提供知识库重建接口：

```bash
curl -sS \
  -X POST http://127.0.0.1:8000/knowledge/rebuild \
  | python -m json.tool --no-ensure-ascii
```

---

## 15. 查看容器内部文件

### 查看 Docker 配置

```bash
./scripts/dc.sh exec api \
  grep -A 12 '^\[embedding\]' /app/config.toml
```

### 查看知识文档

```bash
./scripts/dc.sh exec api \
  ls -lah /app/data/knowledge
```

### 查看 FAISS 索引

```bash
./scripts/dc.sh exec api \
  ls -lah /app/faiss_index
```

### 查看 Embedding 模型

```bash
./scripts/dc.sh exec api \
  ls -lah /models/bge-small-zh-v1.5
```

### 查看容器 Python 版本

```bash
./scripts/dc.sh exec api python --version
```

---

## 16. 常见问题

### 16.1 缺少 `EMBEDDING_MODEL_PATH`

错误：

```text
required variable EMBEDDING_MODEL_PATH is missing a value
```

原因：

直接执行了：

```bash
docker compose ...
```

处理：

```bash
./scripts/dc.sh ps
```

或：

```bash
./scripts/dc.sh logs --tail=100 api
```

日常统一使用 `scripts/dc.sh`。

---

### 16.2 Docker Hub 访问超时

错误：

```text
failed to fetch oauth token
i/o timeout
```

先重试：

```bash
docker pull python:3.11-slim
```

仍然失败时检查：

```text
Docker Desktop
→ Settings
→ Resources
→ Proxies
```

根据网络情况选择：

- System proxy；
- No proxy；
- Manual configuration。

修改后重启 Docker Desktop。

---

### 16.3 pip 依赖冲突

项目 Docker 镜像使用：

```text
requirements-docker.txt
```

Docker 运行依赖中不安装 Gradio。

原因是部分 Gradio 版本和 FastAPI 对 Starlette 的版本要求可能冲突。

本地开发环境仍然使用：

```text
requirements.txt
```

`scripts/rag_debug.py` 在本地 Python 环境运行，不放入 Docker 服务。

---

### 16.4 上游模型接口断开

日志可能出现：

```text
openai.APIConnectionError
httpx.RemoteProtocolError
Server disconnected without sending a response
```

通常表示上游大语言模型服务临时断开。

处理：

1. 重新发送请求；
2. 检查 API Key；
3. 检查模型名；
4. 检查模型服务地址；
5. 查看 API 日志；
6. 等待上游服务恢复。

查看日志：

```bash
./scripts/dc.sh logs --tail=150 api
```

---

### 16.5 Web 返回 500

Web 日志可能显示：

```text
POST /api/chat HTTP/1.1 500
```

Web 服务只是转发请求，真正错误通常在 API 或模型接口中。

查看：

```bash
./scripts/dc.sh logs --tail=150 web
./scripts/dc.sh logs --tail=150 api
```

---

### 16.6 API 返回 502

API 日志可能显示：

```text
POST /chat HTTP/1.1 502 Bad Gateway
```

常见原因：

- 上游模型服务断开；
- 上游模型服务超时；
- API Key 或模型配置错误；
- 网络临时异常。

先查看：

```bash
./scripts/dc.sh logs --tail=150 api
```

---

### 16.7 端口被占用

检查端口：

```bash
ss -ltnp | grep -E ':8000|:8001'
```

停止本地占用：

```bash
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8001/tcp 2>/dev/null || true
```

重新启动：

```bash
./scripts/dc.sh up -d
```

---

### 16.8 容器反复重启

查看状态：

```bash
./scripts/dc.sh ps
```

查看日志：

```bash
./scripts/dc.sh logs --tail=200 api
./scripts/dc.sh logs --tail=200 web
```

常见原因：

- 配置文件格式错误；
- 模型目录挂载失败；
- Python 依赖缺失；
- 应用启动时报错；
- 容器没有文件写入权限。

---

### 16.9 Docker 构建占用空间过大

查看 Docker 磁盘占用：

```bash
docker system df
```

清理未使用的构建缓存：

```bash
docker builder prune
```

清理未使用的容器、网络和镜像：

```bash
docker system prune
```

执行清理前应仔细阅读 Docker 的提示，避免删除仍然需要的镜像。

---

## 17. 数据持久化和备份

主要持久化数据：

```text
data/chat_history.db
data/user_memory.db
data/knowledge/
faiss_index/
```

SQLite 运行期间可能出现：

```text
data/chat_history.db-shm
data/chat_history.db-wal
data/user_memory.db-shm
data/user_memory.db-wal
```

这些是 SQLite WAL 模式的运行时文件，属于正常现象。

备份前建议先停止容器：

```bash
./scripts/dc.sh stop
```

然后复制数据：

```bash
mkdir -p "$HOME/ai/agentic-rag-data-backup"

cp -a data \
  "$HOME/ai/agentic-rag-data-backup/"

cp -a faiss_index \
  "$HOME/ai/agentic-rag-data-backup/"
```

---

## 18. Git 安全检查

以下文件禁止提交：

```text
config.toml
config.docker.toml
data/chat_history.db
data/chat_history.db-shm
data/chat_history.db-wal
data/user_memory.db
data/user_memory.db-shm
data/user_memory.db-wal
```

检查：

```bash
git check-ignore -v \
  config.toml \
  config.docker.toml \
  data/chat_history.db \
  data/chat_history.db-shm \
  data/chat_history.db-wal \
  data/user_memory.db
```

查看即将提交的文件：

```bash
git status --short
```

确认暂存区没有敏感文件：

```bash
git diff --cached --name-only | grep -E \
'config\.toml$|config\.docker\.toml$|\.db(-shm|-wal)?$' \
&& echo "警告：发现不应提交的文件" \
|| echo "敏感配置和数据库未进入暂存区"
```

---

## 19. 公网部署前的安全要求

当前 Docker Compose 方案主要用于：

- 本地学习；
- 功能演示；
- 作品展示；
- 开发测试。

不建议直接暴露到公网。

公网部署前至少需要增加：

- HTTPS；
- API 身份认证；
- 请求频率限制；
- 更严格的 CORS；
- 反向代理；
- 日志脱敏；
- 密钥管理；
- 容器资源限制；
- 数据备份与恢复；
- 服务监控与告警；
- 文件上传安全检查。

---

## 20. 日常命令速查

进入项目：

```bash
cd "$HOME/ai/ai agent"
```

启动项目：

```bash
./scripts/dc.sh up -d
```

查看状态：

```bash
./scripts/dc.sh ps
```

查看日志：

```bash
./scripts/dc.sh logs -f api web
```

停止项目：

```bash
./scripts/dc.sh stop
```

完整关闭：

```bash
./scripts/dc.sh down
```

重启容器：

```bash
./scripts/dc.sh restart
```

修改代码后重新构建：

```bash
./scripts/dc.sh up --build -d
```

修改依赖后完全重建：

```bash
./scripts/dc.sh build --no-cache api web
./scripts/dc.sh up -d
```

查看 API 日志：

```bash
./scripts/dc.sh logs --tail=100 api
```

查看 Web 日志：

```bash
./scripts/dc.sh logs --tail=100 web
```
