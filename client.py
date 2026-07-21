from pathlib import Path
import tomllib

from openai import OpenAI


# client.py 和 config.toml 位于同一目录
CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "config.toml"
)


def load_config():
    """读取项目根目录下的 TOML 配置。"""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "没有找到配置文件："
            f"{CONFIG_PATH}\n"
            "请复制 config.example.toml 为 config.toml，"
            "并填写模型接口配置。"
        )

    with open(
        CONFIG_PATH,
        "rb"
    ) as file:
        return tomllib.load(file)


config = load_config()


# 读取当前使用的模型提供商名称
provider_name = config.get(
    "model_provider",
    "OpenAI"
)


# 读取对应提供商配置
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


# 可在配置中填写，也可以使用默认值
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


client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=timeout,
    max_retries=max_retries
)