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

if [[ ! -f "config.toml" ]]; then
    echo "错误：缺少 config.toml"
    exit 1
fi

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
    echo "请检查 config.toml 中的 embedding.model_path"
    exit 1
fi

export EMBEDDING_MODEL_PATH="$MODEL_DIR"

exec docker compose "$@"
