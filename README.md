# AI Agentic RAG

一个基于大语言模型、工具调用和本地知识库检索构建的 AI Agent 项目。

系统可以根据用户问题自主判断是否需要调用工具，目前支持：

- 普通对话
- 数学表达式计算
- 本地知识库检索
- 同一任务中调用多个工具
- 多轮会话记忆
- Agent 自动评估与报告生成

## 项目特点

### 1. Agent 自主工具选择

项目使用大模型的 Tool Calling 能力，由模型根据用户问题决定：

- 是否调用工具
- 调用哪个工具
- 工具参数是什么
- 是否需要组合多个工具

例如：

```text
用户：计算123*456，并查询知识库说明RAG项目的用途

Agent：
1. 调用 calculator
2. 调用 search_knowledge
3. 综合两个工具的结果生成最终回答
```

### 2. 本地 RAG 知识库

知识库部分使用：

- `bge-small-zh-v1.5` 生成文本向量
- FAISS 保存和检索向量
- PDF、Markdown 等文档作为本地知识来源

基本流程：

```text
本地文档
  ↓
文档加载与切分
  ↓
Embedding 向量化
  ↓
保存到 FAISS
  ↓
根据用户问题检索相关片段
  ↓
大模型结合检索结果生成回答
```

### 3. 多工具执行

Agent 支持在一轮任务中调用多个工具。

当前工具：

| 工具 | 作用 |
|---|---|
| `calculator` | 计算数学表达式 |
| `search_knowledge` | 检索本地 FAISS 知识库 |

工具调用逻辑和工具执行逻辑分别由以下模块管理：

```text
tools_config.py   # 工具 Schema
tools.py          # 工具函数
tool_executor.py  # 工具解析、执行和记录
agent.py          # Agent 循环与任务调度
```

### 4. 会话记忆

`memory.py` 按照 `session_id` 保存不同会话的历史消息，使 Agent 能够处理简单的多轮对话。

### 5. 自动评估

项目包含独立的评估模块，可以检查：

- Agent 是否调用了正确的工具
- 最终回答是否包含预期关键词
- 知识库原始检索结果是否包含正确内容
- 模型请求是否出现超时或系统错误
- 哪些案例需要人工复核

评估结果会保存为：

```text
evaluation/results/
├── evaluation_时间.json
└── latest_report.md
```

其中 `latest_report.md` 可以直接在 GitHub 中查看。

---

## 项目结构

```text
.
├── agent.py
├── client.py
├── main.py
├── memory.py
├── prompt.py
│
├── tools.py
├── tools_config.py
├── tool_executor.py
│
├── embedding.py
├── knowledge_base.py
│
├── data/
│   └── knowledge/
│       ├── AI Agent资料.pdf
│       ├── Prompt Engineering.pdf
│       ├── agents学习路线.md
│       └── 项目文档.pdf
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── evaluation/
│   ├── run_evaluation.py
│   ├── test_cases.json
│   └── results/
│       └── latest_report.md
│
├── config.example.toml
├── requirements.txt
└── README.md
```

---

## 环境要求

推荐环境：

```text
Python 3.13
```

安装项目依赖：

```bash
python -m pip install -r requirements.txt
```

主要依赖：

```text
openai
faiss-cpu
langchain-community
langchain-huggingface
langchain-text-splitters
sentence-transformers
pypdf
```

---

## 模型与配置

项目使用兼容 OpenAI API 格式的模型服务。

复制示例配置：

```bash
cp config.example.toml config.toml
```

然后填写自己的模型配置和 API Key。

示例：

```toml
model_provider = "OpenAI"
model = "gpt-5.6-sol"

[model_providers.OpenAI]
name = "OpenAI"
api_key = "your-api-key"
base_url = "https://your-api-endpoint/v1"
model = "gpt-5.6-sol"
requires_openai_auth = false
```

注意：

- 不要将真实 API Key 提交到 GitHub
- `config.toml` 应加入 `.gitignore`
- 仓库只保留不含密钥的 `config.example.toml`

---

## Embedding 模型

当前知识库使用：

```text
bge-small-zh-v1.5
```

请下载模型并在 `tools.py` 中配置实际模型路径：

```python
model_dir = "/path/to/bge-small-zh-v1.5"
```

当前版本使用本地模型，不会在每次检索时调用远程 Embedding API。

---

## 构建知识库

把知识文档放入：

```text
data/knowledge/
```

然后运行：

```bash
python embedding.py
```

程序会读取文档、切分文本、生成向量，并将 FAISS 索引保存到：

```text
faiss_index/
```

---

## 启动 Agent

在项目根目录运行：

```bash
python main.py
```

启动后：

```text
===================
AI Agent启动
输入 exit 或 quit 退出
===================
```

示例问题：

```text
你好
```

```text
计算123*456
```

```text
请查询知识库，说明RAG项目的用途和基本流程
```

```text
计算123*456，并查询知识库说明RAG项目的用途和基本流程
```

退出：

```text
exit
```

---

## 运行评估

执行：

```bash
python evaluation/run_evaluation.py
```

评估内容包括：

```text
工具选择检查
回答关键词检查
知识库检索内容检查
系统错误统计
人工复核标记
```

评估结束后会生成：

```text
evaluation/results/evaluation_时间.json
evaluation/results/latest_report.md
```

汇总结果示例：

```text
测试数量: 4
正常完成: 4
自动通过: 4
自动未通过: 0
系统错误: 0
模型评估通过率: 100.0%
测试执行成功率: 100.0%
```

评估通过率只代表当前验证集上的结果，不代表系统在所有问题上的真实准确率。

---

## Agent 执行流程

```text
用户问题
   ↓
加入会话历史
   ↓
大模型判断是否调用工具
   ↓
┌─────────────────────┐
│ 不调用工具          │ → 直接生成回答
└─────────────────────┘

┌─────────────────────┐
│ 调用一个或多个工具  │
└─────────────────────┘
   ↓
ToolExecutor 执行工具
   ↓
工具结果返回给大模型
   ↓
模型生成最终回答
```

为防止模型持续调用工具，Agent 设置了最大执行步数限制。

---

## 当前限制

当前版本仍有以下可优化方向：

- 第三方模型接口可能偶发超时
- FAISS 检索暂未设置相关性阈值
- 会话记忆只保存在内存中，程序退出后不会持久化
- 计算工具基于 AST 白名单解析，避免直接使用 eval 执行任意代码
- 本地模型路径目前需要手动配置
- 验证集数量较少，评估结果只能作为阶段性参考

---

## 后续优化方向

- 增加网页搜索、文件读取等工具
- 增加知识库相关性阈值
- 增加对话记录持久化
- 对有副作用的工具增加幂等控制
- 扩充 Bad Case 和自动验证集
- 增加 LLM-as-a-Judge 多维评估
- 使用 FastAPI 提供 HTTP 服务
- 增加简单的 Web 交互界面

---

## 项目定位

本项目主要用于学习和展示以下能力：

- OpenAI-compatible API 调用
- Tool Calling
- Agent 循环设计
- 多工具任务执行
- RAG 知识库构建
- FAISS 向量检索
- 多轮会话管理
- 自动化评估
- 错误处理与工程模块化