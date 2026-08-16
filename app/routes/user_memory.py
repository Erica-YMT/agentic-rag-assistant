"""长期用户记忆查询和删除接口。"""

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query

from app.memory.user_memory import UserMemoryStore
from app.auth.router import get_current_user


router = APIRouter(
    prefix="/memory",
    tags=["长期记忆"],
    dependencies=[Depends(get_current_user)],
)

@lru_cache(maxsize=1)
def get_store() -> UserMemoryStore:
    return UserMemoryStore()


@router.get("")
def list_memories(
    limit: int = Query(default=100, ge=1, le=500),
    current_user = Depends(get_current_user),
):
    return {
        "items": get_store().list(
            user_id=str(current_user["id"]),
            limit=limit,
        ),
    }


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: int,
    current_user = Depends(get_current_user),
):
    deleted = get_store().delete(
        memory_id,
        user_id=str(current_user["id"]),
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="没有找到该记忆",
        )

    return {
        "ok": True,
        "memory_id": memory_id,
    }


# RBAC_AUTH_V1
# MULTI_USER_ISOLATION_V1
