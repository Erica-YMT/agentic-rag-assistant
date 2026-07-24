from pathlib import Path
import tomllib


# =========================
# 配置文件路径
# =========================

CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "config.toml"
)


def load_config():
    """
    读取项目根目录下的 TOML 配置。

    这个模块只负责配置读取，
    不创建模型客户端，也不加载知识库。
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "没有找到配置文件："
            f"{CONFIG_PATH}\n"
            "请复制 config.example.toml 为 config.toml，"
            "并填写模型与知识库配置。"
        )

    with open(
        CONFIG_PATH,
        "rb"
    ) as file:
        return tomllib.load(file)


# 项目统一使用的配置字典
config = load_config()
