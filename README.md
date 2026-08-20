# Agentic RAG Assistant

> Production-oriented LLM Agent + Advanced RAG application for
> enterprise knowledge-base question answering.

## Overview

Agentic RAG Assistant 是一个面向中文知识问答场景的工程化 AI Agent 系统。

项目围绕 **LLM Agent、Tool Calling、Advanced
RAG、MCP、Memory、Evaluation 和 Observability**
构建，不仅实现基础的"检索增强生成"，还进一步加入复杂问题规划、检索纠错、多用户隔离、工程部署与运行监控能力。

核心目标：

-   让 LLM 具备自主决策和工具调用能力
-   让知识库问答具备更高召回率和可靠性
-   构建接近生产环境的 Agent 应用架构

------------------------------------------------------------------------

# Architecture Overview

``` text
Browser
   |
FastAPI + Uvicorn
   |
Agent Session Service
   |
Agent Core
   |
   +---- LLM
   |
   +---- Tool Executor
   |        |
   |        + Calculator
   |        + Web Search
   |        + Knowledge Search
   |        + MCP Tools
   |
   |
   +---- Advanced RAG
            |
            + Hybrid Retrieval
            |       |
            |       + BM25
            |       + Dense Vector Search
            |       + RRF Fusion
            |
            + Corrective RAG
            |
            + Complex RAG
            |
            + Hierarchical Chunking
            + Auto Merge


Storage:
PostgreSQL
Redis
FAISS / Milvus


Observability:
Prometheus
Grafana
LangSmith
```

------------------------------------------------------------------------

# Core Features

## 1. Agent System

实现完整 Agent Loop：

``` text
User Query
    ↓
LLM Decision
    ↓
tool_calls
    ↓
Tool Executor
    ↓
Tool Result
    ↓
LLM Continue Reasoning
    ↓
Final Answer
```

支持：

-   Function Calling
-   Tool Schema 注册
-   Tool Executor
-   Tool Timeout
-   Retry
-   Fallback
-   Tool Governance

------------------------------------------------------------------------

## 2. Advanced RAG Pipeline

支持多种检索策略：

### Hybrid Retrieval

结合：

-   BM25 Sparse Retrieval
-   Dense Vector Retrieval
-   RRF Fusion

### Corrective RAG

流程：

``` text
Query
 ↓
Retrieve
 ↓
Evidence Grading
 ↓
不足
 ↓
Query Rewrite
 ↓
Retrieve Again
 ↓
Final Context
```

### Complex RAG

针对复杂问题：

``` text
Complex Query
 ↓
Planner
 ↓
Sub Questions
 ↓
Parallel Retrieval
 ↓
Evidence Merge
 ↓
Coverage Evaluation
 ↓
Answer Generation
```

### Hierarchical Retrieval

支持：

-   Parent-Child Chunk
-   Auto Merge
-   Context Expansion

------------------------------------------------------------------------

# Vector Backend

支持：

## FAISS

默认稳定后端。

## Milvus

完整接入：

-   Collection 管理
-   Schema 校验
-   Full Synchronization
-   Private Knowledge Isolation

架构：

``` text
Document
   ↓
Chunk
   ↓
Embedding
   ↓
Milvus Collection
   ↓
Vector Search
```

------------------------------------------------------------------------

# Tool Ecosystem

Local Tools:

-   Knowledge Search
-   Calculator
-   Web Search

MCP Integration:

-   MCP Client
-   MCP Server
-   Streamable HTTP
-   Docker Sidecar

支持：

``` text
Agent
 ↓
ToolExecutor
 ↓
MCP Client
 ↓
MCP Server
 ↓
External Capability
```

------------------------------------------------------------------------

# Memory & User System

实现：

-   JWT Authentication
-   User Isolation
-   Session Management
-   Chat Memory
-   Explicit Long-term Memory

数据：

``` text
PostgreSQL
    |
    + Users
    + Sessions
    + Messages
    + Knowledge Documents


Redis
    |
    + Chat Cache
```

------------------------------------------------------------------------

# Engineering Features

## Concurrency Control

支持：

-   Session Lock
-   Global Semaphore

保证：

-   同一会话顺序执行
-   系统并发保护

## Streaming

支持：

-   Agent Event Streaming
-   NDJSON Response
-   Answer Delta

## Monitoring

集成：

-   Prometheus Metrics
-   Grafana Dashboard
-   LangSmith Tracing

------------------------------------------------------------------------

# Evaluation

设计 Agent 回归测试：

验证：

-   Tool Routing
-   Tool Parameters
-   Retrieval Pipeline
-   Multi-tool Workflow

当前验证：

``` text
Test Cases: 4

Execution Success Rate:
100%

Validated:

✓ Normal Conversation
✓ Calculator Tool Calling
✓ Knowledge Retrieval
✓ Multi-tool Workflow
```

说明：

评估用于验证系统行为和链路稳定性，不代表模型真实准确率。

------------------------------------------------------------------------

# Project Structure

``` text
app/
├── agent/
├── auth/
├── core/
├── db/
├── integrations/
├── memory/
├── routes/
├── services/

rag/
├── knowledge_base.py
├── retriever.py
├── corrective_rag.py
├── rag_graph.py
├── milvus_store.py

evaluation/
├── agent_harness.py
├── run_evaluation.py

monitoring/
├── prometheus/
└── grafana/
```

------------------------------------------------------------------------

# Deployment

启动：

``` bash
docker compose up -d
```

服务：

  Service      Port
  ------------ -------
  API          8000
  Web          8001
  Grafana      3000
  Milvus       19530
  Prometheus   9090

------------------------------------------------------------------------

# Technical Stack

  Category     Technology
  ------------ -----------------------------------
  Backend      FastAPI
  Server       Uvicorn
  LLM          OpenAI-compatible API
  Agent        Custom Agent Loop
  RAG          Hybrid / Corrective / Complex RAG
  Vector DB    FAISS / Milvus
  Database     PostgreSQL
  Cache        Redis
  Monitoring   Prometheus + Grafana
  Protocol     MCP
  Deployment   Docker Compose

------------------------------------------------------------------------

# Project Positioning

该项目不是简单的大模型 API 调用 Demo，而是一个完整的：

**LLM Agent Application Engineering Project**

覆盖：

-   模型调用
-   Agent 决策
-   工具系统
-   知识检索
-   数据存储
-   用户系统
-   监控部署
-   自动评估
