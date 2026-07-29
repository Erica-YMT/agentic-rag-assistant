"""长期用户记忆查询和删除接口。"""

from fastapi import APIRouter, HTTPException, Query

from user_memory import UserMemoryStore


router = APIRouter(
    prefix="/memory",
    tags=["长期记忆"],
)

store = UserMemoryStore()


@router.get("")
def list_memories(
    limit: int = Query(default=100, ge=1, le=500),
):
    return {
        "items": store.list(limit=limit),
    }


@router.delete("/{memory_id}")
def delete_memory(memory_id: int):
    deleted = store.delete(memory_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="没有找到该记忆",
        )

    return {
        "ok": True,
        "memory_id": memory_id,
    }
