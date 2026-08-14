from openai import OpenAI

from config import config


# =========================
# 读取模型提供商
# =========================

provider_name = config.get(
    "model_provider",
    "OpenAI"
)


provider_config = (
    config
    .get("model_providers", {})
    .get(provider_name)
)


if not provider_config:
    raise ValueError(
        "配置文件中不存在模型提供商："
        f"{provider_name}"
    )


# =========================
# 读取模型配置
# =========================

api_key = provider_config.get(
    "api_key"
)

base_url = provider_config.get(
    "base_url"
)

model_name = (
    provider_config.get("model")
    or config.get("model")
)


if not api_key:
    raise ValueError(
        "config.toml 中缺少 api_key"
    )

if not base_url:
    raise ValueError(
        "config.toml 中缺少 base_url"
    )

if not model_name:
    raise ValueError(
        "config.toml 中缺少 model"
    )


# =========================
# 请求配置
# =========================

timeout = float(
    provider_config.get(
        "timeout",
        30.0
    )
)

max_retries = int(
    provider_config.get(
        "max_retries",
        1
    )
)


# =========================
# 创建大模型客户端
# =========================

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=timeout,
    max_retries=max_retries
)
