#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

PROJECT_ROOT="$(
    cd "$SCRIPT_DIR/.."
    pwd
)"

cd "$PROJECT_ROOT"

echo "=== Agentic RAG Docker 启动 ==="

# 检查必需文件
required_files=(
    "Dockerfile"
    "docker-compose.yml"
    "requirements.txt"
    "config.toml"
    "config.docker.toml"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "错误：缺少文件 $file"
        exit 1
    fi
done

# 从本地 config.toml 读取 Embedding 模型目录
MODEL_DIR="$(
python - <<'PY'
from pathlib import Path
import tomllib

with open("config.toml", "rb") as file:
    config = tomllib.load(file)

raw_path = str(config["embedding"]["model_path"]).strip()
path = Path(raw_path).expanduser()

if not path.is_absolute():
    path = Path.cwd() / path

print(path.resolve())
PY
)"

# 如果配置中不是有效目录，在 ModelScope 缓存中查找
if [[ ! -d "$MODEL_DIR" ]]; then
    MODEL_DIR="$(
        find "$HOME/.cache/modelscope" \
            -type d \
            -name "bge-small-zh-v1.5" \
            -print \
            -quit \
            2>/dev/null || true
    )"
fi

if [[ -z "$MODEL_DIR" || ! -d "$MODEL_DIR" ]]; then
    echo "错误：没有找到 bge-small-zh-v1.5 模型目录"
    echo "请检查 [config.toml] 中的 [embedding].model_path"
    exit 1
fi

export EMBEDDING_MODEL_PATH="$MODEL_DIR"

mkdir -p data/knowledge
mkdir -p faiss_index

echo "Embedding 模型目录："
echo "$EMBEDDING_MODEL_PATH"
echo

# 防止本地服务占用 Docker 端口
fuser -k 8000/tcp 2>/dev/null || true
fuser -k 8001/tcp 2>/dev/null || true

echo "正在检查 [docker-compose.yml]……"
docker compose config >/dev/null

echo "Compose 配置检查通过。"
echo "正在构建并启动 API 和 Web 容器……"

docker compose up --build -d

echo
echo "=== 容器状态 ==="
docker compose ps

echo
echo "网页：http://127.0.0.1:8001"
echo "API 文档：http://127.0.0.1:8000/docs"
echo "健康检查：http://127.0.0.1:8000/health"
