from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.toml"


def load_config() -> dict:
    """读取项目的 TOML 配置。"""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"没有找到配置文件：{CONFIG_PATH}\n"
            "请先创建 config.toml。"
        )

    with CONFIG_PATH.open("rb") as file:
        return tomllib.load(file)


def resolve_project_path(path_value: str) -> Path:
    """将相对路径转换成以项目根目录为基准的绝对路径。"""

    path = Path(path_value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_documents(data_path: Path) -> list:
    """读取知识库目录中的 PDF、Markdown 和 TXT 文件。"""

    if not data_path.exists():
        raise FileNotFoundError(
            f"知识库目录不存在：{data_path}"
        )

    documents = []
    supported_suffixes = {".pdf", ".md", ".txt"}

    file_paths = sorted(
        path
        for path in data_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_suffixes
    )

    if not file_paths:
        raise ValueError(
            f"目录中没有找到 PDF、Markdown 或 TXT 文件：{data_path}"
        )

    for file_path in file_paths:
        suffix = file_path.suffix.lower()

        print(f"正在读取：{file_path.name}")

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(
                    str(file_path)
                )
            else:
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                    autodetect_encoding=True,
                )

            loaded_documents = loader.load()

        except Exception as exc:
            print(
                f"跳过无法读取的文件：{file_path.name}\n"
                f"原因：{exc}"
            )
            continue

        relative_path = file_path.relative_to(
            PROJECT_ROOT
        )

        for document in loaded_documents:
            document.metadata["file_name"] = (
                file_path.name
            )
            document.metadata["relative_path"] = str(
                relative_path
            )
            document.metadata["file_type"] = suffix

        documents.extend(loaded_documents)

    if not documents:
        raise RuntimeError(
            "文件虽然存在，但没有成功读取出任何文本。"
        )

    return documents


def split_documents(
    documents: list,
    chunk_size: int,
    chunk_overlap: int,
) -> list:
    """把长文档切分成适合检索的小块。"""

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap 必须小于 chunk_size。"
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            "",
        ],
        length_function=len,
    )

    return text_splitter.split_documents(
        documents
    )


def build_index() -> None:
    """构建并保存 FAISS 知识库索引。"""

    config = load_config()
    embedding_config = config.get(
        "embedding",
        {},
    )

    model_path_value = embedding_config.get(
        "model_path"
    )

    if not model_path_value:
        raise ValueError(
            "config.toml 中缺少 "
            "[embedding].model_path"
        )

    data_path = resolve_project_path(
        embedding_config.get(
            "data_path",
            "data/knowledge",
        )
    )

    index_path = resolve_project_path(
        embedding_config.get(
            "index_path",
            "faiss_index",
        )
    )

    model_path = resolve_project_path(
        model_path_value
    )

    chunk_size = int(
        embedding_config.get(
            "chunk_size",
            600,
        )
    )

    chunk_overlap = int(
        embedding_config.get(
            "chunk_overlap",
            100,
        )
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"没有找到 Embedding 模型：{model_path}"
        )

    print("\n========== 构建个人知识库 ==========")
    print(f"文档目录：{data_path}")
    print(f"模型路径：{model_path}")
    print(f"索引目录：{index_path}")
    print(
        f"切分设置：chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}"
    )

    documents = load_documents(data_path)

    print(
        f"\n成功读取 Document 数量："
        f"{len(documents)}"
    )

    chunks = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print(
        f"文本切分后 Chunk 数量："
        f"{len(chunks)}"
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name=str(model_path),
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    print("\n正在生成文本向量……")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    # 删除旧索引，避免旧文件残留。
    if index_path.exists():
        shutil.rmtree(index_path)

    index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(
        str(index_path)
    )

    print("\n========== 构建完成 ==========")
    print(f"原始 Document：{len(documents)}")
    print(f"文本 Chunk：{len(chunks)}")
    print(f"FAISS 索引：{index_path}")
    print(
        "生成文件：index.faiss、index.pkl"
    )


if __name__ == "__main__":
    build_index()