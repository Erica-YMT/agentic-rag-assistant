from __future__ import annotations

import json
from email import policy
from email.parser import BytesParser
from html import unescape
import shutil
import torch
import tomllib
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


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


def build_index_manifest(settings: dict) -> dict:
    """生成决定 Chunk/Embedding 兼容性的索引配置快照。"""

    embedding = settings.get("embedding", {})
    hierarchical = settings.get("hierarchical_chunking", {})

    return {
        "version": 1,
        "embedding_model_path": str(embedding.get("model_path", "")),
        "normalize_embeddings": True,
        "hierarchical_enabled": bool(hierarchical.get("enabled", True)),
        "chunk_size": int(embedding.get("chunk_size", 600)),
        "chunk_overlap": int(embedding.get("chunk_overlap", 100)),
        "parent_chunk_size": int(hierarchical.get("parent_chunk_size", 1200)),
        "parent_chunk_overlap": int(hierarchical.get("parent_chunk_overlap", 120)),
        "child_chunk_size": int(hierarchical.get("child_chunk_size", 400)),
        "child_chunk_overlap": int(hierarchical.get("child_chunk_overlap", 80)),
    }


def save_index_manifest(index_path: Path, settings: dict) -> None:
    """保存索引配置快照，增量更新前用它判断是否必须全量重建。"""

    payload = build_index_manifest(settings)
    path = Path(index_path) / "index_manifest.json"
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".md", ".txt", ".xlsx", ".eml", ".case"}


def _load_structured_file(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".xlsx":
        try:
            import openpyxl
        except ImportError as error:
            raise RuntimeError(
                "XLSX 解析依赖未安装，请安装 openpyxl 后重试。"
            ) from error
        workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        documents = []
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                values = [str(value).strip() for value in row if value is not None and str(value).strip()]
                if values:
                    rows.append(" | ".join(values))
            if rows:
                documents.append(Document(page_content=f"Sheet: {sheet.title}\n" + "\n".join(rows), metadata={"sheet": sheet.title}))
        return documents
    if suffix == ".eml":
        with file_path.open("rb") as stream:
            message = BytesParser(policy=policy.default).parse(stream)
        parts = [f"Subject: {message.get('subject', '')}", f"From: {message.get('from', '')}", f"To: {message.get('to', '')}"]
        body = message.get_body(preferencelist=("plain", "html"))
        if body is not None:
            parts.append(unescape(body.get_content()))
        return [Document(page_content="\n".join(parts), metadata={"email_subject": message.get("subject", "")})]
    if suffix == ".case":
        return [Document(page_content=file_path.read_text(encoding="utf-8"), metadata={})]
    return []


def load_document_file(file_path: Path) -> list:
    """读取单个知识文档，并统一补充稳定的来源 metadata。"""

    file_path = Path(file_path).resolve()
    suffix = file_path.suffix.lower()

    if not file_path.is_file():
        raise FileNotFoundError(f"知识文档不存在：{file_path}")

    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError(f"不支持的知识文档类型：{suffix}")

    print(f"正在读取：{file_path.name}")

    if suffix in {".xlsx", ".eml", ".case"}:
        loaded_documents = _load_structured_file(file_path)
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        loaded_documents = loader.load()
    else:
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
            autodetect_encoding=True,
        )
        loaded_documents = loader.load()
    try:
        relative_path = file_path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative_path = Path(file_path.name)

    for document in loaded_documents:
        document.metadata["file_name"] = file_path.name
        document.metadata["relative_path"] = relative_path.as_posix()
        document.metadata["file_type"] = suffix

    return loaded_documents


def load_documents(data_path: Path) -> list:
    """读取知识库目录中的 PDF、Markdown 和 TXT 文件。"""

    if not data_path.exists():
        raise FileNotFoundError(
            f"知识库目录不存在：{data_path}"
        )

    documents = []

    file_paths = sorted(
        path
        for path in data_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
    )

    if not file_paths:
        raise ValueError(
            f"目录中没有找到 PDF、Markdown 或 TXT 文件：{data_path}"
        )

    for file_path in file_paths:
        try:
            documents.extend(load_document_file(file_path))
        except Exception as exc:
            print(
                f"跳过无法读取的文件：{file_path.name}\n"
                f"原因：{exc}"
            )

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


def build_index(
    *,
    data_path_override: str | Path | None = None,
    index_path_override: str | Path | None = None,
) -> None:
    """
    构建并保存 FAISS 知识库索引。

    hierarchical_chunking.enabled=true：

        原文
        -> Parent Chunk
        -> Child Chunk
        -> Child 进入 FAISS
        -> Parent 保存 parent_store.json

    hierarchical_chunking.enabled=false：

        保留原来的一层 Chunk 方案。
    """

    config = load_config()

    embedding_config = (
        config.get(
            "embedding",
            {},
        )
    )

    hierarchical_config = (
        config.get(
            "hierarchical_chunking",
            {},
        )
    )

    model_path_value = (
        embedding_config.get(
            "model_path"
        )
    )

    if not model_path_value:
        raise ValueError(
            "config.toml 中缺少 "
            "[embedding].model_path"
        )

    configured_data_path = resolve_project_path(
        embedding_config.get(
            "data_path",
            "data/knowledge/public",
        )
    )

    # Backward-safe migration rule: an old config that still points at
    # data/knowledge must never recursively absorb data/knowledge/users/*.
    legacy_root = (PROJECT_ROOT / "data" / "knowledge").resolve()
    if data_path_override is not None:
        data_path = resolve_project_path(str(data_path_override))
    elif configured_data_path == legacy_root:
        data_path = (legacy_root / "public").resolve()
    else:
        data_path = configured_data_path

    if index_path_override is not None:
        index_path = resolve_project_path(str(index_path_override))
    else:
        index_path = resolve_project_path(
            embedding_config.get(
                "index_path",
                "faiss_index",
            )
        )

    model_path = (
        resolve_project_path(
            model_path_value
        )
    )

    embedding_batch_size = max(
        1,
        int(
            embedding_config.get(
                "batch_size",
                32,
            )
        ),
    )

    embedding_device = str(
        embedding_config.get(
            "device",
            "auto",
        )
    ).strip().lower()

    if embedding_device not in {
        "auto",
        "cpu",
        "cuda",
    }:
        raise ValueError(
            "[embedding].device "
            "只支持 auto / cpu / cuda"
        )

    if embedding_device == "auto":
        resolved_embedding_device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    elif embedding_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[embedding].device=cuda，"
                "但当前运行环境无法使用 CUDA"
            )

        resolved_embedding_device = "cuda"

    else:
        resolved_embedding_device = "cpu"

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

    hierarchical_enabled = bool(
        hierarchical_config.get(
            "enabled",
            True,
        )
    )

    parent_chunk_size = int(
        hierarchical_config.get(
            "parent_chunk_size",
            1200,
        )
    )

    parent_chunk_overlap = int(
        hierarchical_config.get(
            "parent_chunk_overlap",
            120,
        )
    )

    child_chunk_size = int(
        hierarchical_config.get(
            "child_chunk_size",
            400,
        )
    )

    child_chunk_overlap = int(
        hierarchical_config.get(
            "child_chunk_overlap",
            80,
        )
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "没有找到 Embedding 模型："
            f"{model_path}"
        )

    print()
    print(
        "========== 构建个人知识库 =========="
    )
    print(
        f"文档目录：{data_path}"
    )
    print(
        f"模型路径：{model_path}"
    )
    print(
        f"索引目录：{index_path}"
    )

    if hierarchical_enabled:

        print(
            "切分模式："
            "Parent-Child Hierarchical"
        )

        print(
            "Parent："
            f"size={parent_chunk_size}, "
            f"overlap="
            f"{parent_chunk_overlap}"
        )

        print(
            "Child："
            f"size={child_chunk_size}, "
            f"overlap="
            f"{child_chunk_overlap}"
        )

    else:

        print(
            "切分模式："
            "Single Chunk"
        )

        print(
            "切分设置："
            f"chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    documents = (
        load_documents(
            data_path
        )
    )

    print(
        "\n成功读取 Document 数量："
        f"{len(documents)}"
    )

    parent_documents = []
    parent_store = {}

    if hierarchical_enabled:

        from rag.hierarchical_chunks import (
            build_hierarchical_chunks,
            save_parent_store,
        )

        (
            parent_documents,
            chunks,
            parent_store,
        ) = build_hierarchical_chunks(
            documents=documents,

            parent_chunk_size=(
                parent_chunk_size
            ),

            parent_chunk_overlap=(
                parent_chunk_overlap
            ),

            child_chunk_size=(
                child_chunk_size
            ),

            child_chunk_overlap=(
                child_chunk_overlap
            ),
        )

        print(
            "Parent Chunk 数量："
            f"{len(parent_documents)}"
        )

        print(
            "Child Chunk 数量："
            f"{len(chunks)}"
        )

    else:

        chunks = split_documents(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=(
                chunk_overlap
            ),
        )

        print(
            "文本 Chunk 数量："
            f"{len(chunks)}"
        )

    if not chunks:
        raise RuntimeError(
            "没有可写入 FAISS 的 Chunk"
        )

    embedding_model = (
        HuggingFaceEmbeddings(
            model_name=str(
                model_path
            ),
            model_kwargs={
                "device":
                    resolved_embedding_device,
            },
            encode_kwargs={
                "normalize_embeddings":
                    True,
                "batch_size":
                    embedding_batch_size,
            },
        )
    )

    print(
        "\n正在生成 Child 文本向量……"
        if hierarchical_enabled
        else
        "\n正在生成文本向量……"
    )

    vector_store = (
        FAISS.from_documents(
            documents=chunks,
            embedding=(
                embedding_model
            ),
        )
    )

    # 不直接删除整个目录，
    # 避免 Docker / WSL 下目录锁问题。
    if index_path.exists():

        for item in index_path.iterdir():

            # The default public index lives at faiss_index/.
            # Keep faiss_index/users/ because each user owns an isolated
            # private FAISS index there.
            if (
                index_path_override is None
                and item.name == "users"
                and item.is_dir()
            ):
                continue

            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    else:

        index_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    vector_store.save_local(
        str(
            index_path
        )
    )

    if hierarchical_enabled:

        save_parent_store(
            (
                index_path
                / "parent_store.json"
            ),
            parent_store,
        )

    save_index_manifest(index_path, config)

    print()
    print(
        "========== 构建完成 =========="
    )

    print(
        "原始 Document："
        f"{len(documents)}"
    )

    if hierarchical_enabled:

        print(
            "Parent Chunk："
            f"{len(parent_documents)}"
        )

        print(
            "Child Chunk："
            f"{len(chunks)}"
        )

        print(
            "向量化对象："
            "仅 Child Chunk"
        )

        print(
            "Parent Store："
            f"{index_path / 'parent_store.json'}"
        )

    else:

        print(
            "文本 Chunk："
            f"{len(chunks)}"
        )

    print(
        f"FAISS 索引：{index_path}"
    )

    generated_files = (
        "index.faiss、index.pkl、index_manifest.json"
    )

    if hierarchical_enabled:
        generated_files += (
            "、parent_store.json"
        )

    print(
        "生成文件："
        f"{generated_files}"
    )



if __name__ == "__main__":
    build_index()
