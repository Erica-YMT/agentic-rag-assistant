FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# FAISS 运行时可能需要
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先安装依赖
COPY requirements-docker.txt ./

RUN python -m pip install \
        --timeout 180 \
        --retries 10 \
        --upgrade pip \
    && python -m pip install \
        --timeout 180 \
        --retries 10 \
        --prefer-binary \
        -r requirements-docker.txt

# 再复制项目代码
COPY . ./

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
