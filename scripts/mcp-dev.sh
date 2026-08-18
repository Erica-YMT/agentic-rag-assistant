#!/usr/bin/env bash

set -e

cd "$(dirname "$0")/.."

# 项目根目录加入 Python 模块搜索路径
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# MCP 的知识库检索会使用项目 LLM，
# 所以要求提前设置 MODEL_API_KEY。
if [ -z "${MODEL_API_KEY:-}" ]; then
    echo "❌ 未设置 MODEL_API_KEY"
    echo
    echo '请先执行：'
    echo 'read -s -p "请输入 MODEL_API_KEY: " MODEL_API_KEY'
    echo 'echo'
    echo 'export MODEL_API_KEY'
    exit 1
fi

echo "======================================"
echo " Agentic RAG Assistant MCP Dev"
echo "======================================"
echo
echo "MCP Tools:"
echo "  - calculator"
echo "  - weather"
echo "  - search_knowledge"
echo

npx --yes @modelcontextprotocol/inspector@2.2.0 \
  -e MODEL_API_KEY="$MODEL_API_KEY" \
  -e PYTHONPATH="$PYTHONPATH" \
  --cwd "$PWD" \
  -- \
  uv run --with mcp==2.0.0 \
  mcp run app/integrations/mcp_server.py
