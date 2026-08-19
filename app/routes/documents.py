"""Authenticated public/private knowledge-document APIs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.auth.router import get_current_user
from app.schemas import (
    KnowledgeDocumentDeleteRequest,
    KnowledgeDocumentListResponse,
    KnowledgeDocumentMutationResponse,
)
from app.services.knowledge_service import knowledge_service


router = APIRouter(
    prefix="/documents",
    tags=["知识库文档"],
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _list_payload(current_user: dict) -> dict:
    return knowledge_service.list_documents(current_user)


@router.get(
    "",
    response_model=KnowledgeDocumentListResponse,
    summary="查看公共文档和我的私有文档",
)
def list_documents(
    current_user=Depends(get_current_user),
):
    return _list_payload(current_user)


@router.post(
    "/upload",
    response_model=KnowledgeDocumentMutationResponse,
    summary="上传公共或私有知识文档",
)
async def upload_documents(
    files: list[UploadFile] = File(...),
    rebuild: bool = Query(default=True),
    current_user=Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=422, detail="请选择至少一个文件")

    uploads: list[tuple[str, bytes]] = []
    total_bytes = 0

    for upload in files:
        content = await upload.read()
        total_bytes += len(content)

        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="单次上传文件总大小不能超过 20 MB",
            )

        uploads.append((upload.filename or "", content))

    try:
        saved_rows = knowledge_service.save_documents(
            current_user=current_user,
            uploads=uploads,
        )
        names = [str(row["filename"]) for row in saved_rows]

        elapsed = 0.0
        index_rebuilt = False

        if rebuild:
            elapsed = knowledge_service.index_documents(
                current_user=current_user,
                rows=saved_rows,
            )
            index_rebuilt = True

        role = str(current_user.get("role", "user")).lower()
        target = "公共知识库" if role == "admin" else "我的私有知识库"
        action = "并已更新索引" if index_rebuilt else "，尚未更新索引"

        return {
            "ok": True,
            "message": f"已上传到{target}：{'、'.join(names)}{action}。",
            "index_rebuilt": index_rebuilt,
            "elapsed_seconds": elapsed,
            **_list_payload(current_user),
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post(
    "/rebuild",
    response_model=KnowledgeDocumentMutationResponse,
    summary="重建当前用户可管理的知识库索引",
)
def rebuild_documents(
    current_user=Depends(get_current_user),
):
    try:
        elapsed = knowledge_service.rebuild_for_user(current_user)
        target = (
            "公共知识库"
            if str(current_user.get("role", "")).lower() == "admin"
            else "我的私有知识库"
        )
        return {
            "ok": True,
            "message": f"{target}索引已重建。",
            "index_rebuilt": True,
            "elapsed_seconds": elapsed,
            **_list_payload(current_user),
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post(
    "/delete",
    response_model=KnowledgeDocumentMutationResponse,
    summary="删除我有权限管理的知识文档",
)
def delete_documents(
    request: KnowledgeDocumentDeleteRequest,
    current_user=Depends(get_current_user),
):
    try:
        names, index_rebuilt, elapsed = knowledge_service.delete_documents(
            current_user=current_user,
            document_ids=request.document_ids,
            rebuild=True,
        )

        return {
            "ok": True,
            "message": f"已删除：{'、'.join(names)}。",
            "index_rebuilt": index_rebuilt,
            "elapsed_seconds": elapsed,
            **_list_payload(current_user),
        }

    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
