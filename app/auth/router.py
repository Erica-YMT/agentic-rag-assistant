"""用户注册、登录和当前用户接口。"""

from __future__ import annotations

import re
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from pydantic import (
    BaseModel,
    Field,
)

from app.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.auth.user_store import (
    UserStore,
)
from app.auth.email_verification import (
    EmailDeliveryError,
    email_verification_store,
)


router = APIRouter(
    prefix="/auth",
    tags=["认证"],
)

bearer_scheme = HTTPBearer(
    auto_error=False
)

store = UserStore()

USERNAME_PATTERN = re.compile(
    r"^[a-zA-Z0-9_-]+$"
)


class RegisterRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=32,
    )

    password: str = Field(
        min_length=8,
        max_length=128,
    )
    email: str | None = None


class LoginRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=32,
    )

    password: str = Field(
        min_length=1,
        max_length=128,
    )


class EmailCodeRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)


class EmailLoginRequest(EmailCodeRequest):
    code: str = Field(min_length=6, max_length=6)


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: str
    updated_at: str
    email: str | None = None


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


def _validated_username(
    value: str,
) -> str:
    username = (
        UserStore.normalize_username(
            value
        )
    )

    if (
        not 3 <= len(username) <= 32
        or not USERNAME_PATTERN.fullmatch(
            username
        )
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "用户名只能包含字母、数字、"
                "下划线和横线，长度 3～32"
            ),
        )

    return username


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="登录状态无效或已经过期",
        headers={
            "WWW-Authenticate":
                "Bearer"
        },
    )


def authenticate_access_token(
    token: str,
) -> dict[str, Any]:
    """校验 JWT，并以数据库中的当前用户状态/角色为准。"""

    try:
        payload = decode_access_token(
            str(token)
        )
    except ValueError:
        raise _unauthorized()

    user = store.get_user_by_id(
        payload["user_id"]
    )

    if user is None:
        raise _unauthorized()

    if not bool(user["is_active"]):
        raise HTTPException(
            status_code=403,
            detail="当前账号已被停用",
        )

    return user


def get_current_user(
    credentials:
        HTTPAuthorizationCredentials
        | None
        = Depends(
            bearer_scheme
        ),
) -> dict[str, Any]:

    if (
        credentials is None
        or credentials.scheme.lower()
        != "bearer"
    ):
        raise _unauthorized()

    return authenticate_access_token(
        credentials.credentials
    )


def require_admin(
    current_user:
        dict[str, Any]
        = Depends(
            get_current_user
        ),
) -> dict[str, Any]:
    """RBAC：admin-only dependency。"""

    if str(
        current_user.get("role", "")
    ).lower() != "admin":
        raise HTTPException(
            status_code=403,
            detail="该操作仅管理员可执行",
        )

    return current_user


# RBAC_AUTH_V1


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201,
    summary="注册用户",
)
def register(
    request: RegisterRequest,
):
    username = _validated_username(
        request.username
    )

    if (
        store.get_auth_user_by_username(
            username
        )
        is not None
    ):
        raise HTTPException(
            status_code=409,
            detail="用户名已经存在",
        )

    password_hash = hash_password(
        request.password
    )

    user = store.create_user(
        username=username,
        password_hash=password_hash,
        role="user",
        email=request.email,
    )

    if user is None:
        raise HTTPException(
            status_code=409,
            detail="用户名已经存在",
        )

    return user


@router.post("/request-code")
def request_email_code(request: EmailCodeRequest):
    if not email_verification_store.is_delivery_configured():
        raise HTTPException(
            status_code=503,
            detail="邮箱验证码服务尚未配置 SMTP。",
        )
    try:
        email_verification_store.issue(request.email)
    except EmailDeliveryError as error:
        raise HTTPException(
            status_code=503,
            detail="验证码发送失败：SMTP 服务器不可达或拒绝投递。",
        ) from error
    return {"ok": True, "message": "验证码已发送，请查收邮箱。"}


@router.post("/login-email", response_model=LoginResponse)
def login_email(request: EmailLoginRequest):
    if not email_verification_store.verify(request.email, request.code):
        raise HTTPException(status_code=401, detail="验证码无效或已过期")
    user = store.get_auth_user_by_email(request.email)
    if user is None or not bool(user["is_active"]):
        raise HTTPException(status_code=401, detail="邮箱或验证码错误")
    token, expires_in = create_access_token(user_id=int(user["id"]), username=str(user["username"]), role=str(user["role"]))
    return {"access_token": token, "token_type": "bearer", "expires_in": expires_in, "user": {key: user.get(key) for key in ("id", "username", "email", "role", "is_active", "created_at", "updated_at")}}


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="登录并获取 JWT",
)
def login(
    request: LoginRequest,
):
    username = _validated_username(
        request.username
    )

    user = (
        store.get_auth_user_by_username(
            username
        )
    )

    if (
        user is None
        or not verify_password(
            request.password,
            user["password_hash"],
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={
                "WWW-Authenticate":
                    "Bearer"
            },
        )

    if not bool(
        user["is_active"]
    ):
        raise HTTPException(
            status_code=403,
            detail="当前账号已被停用",
        )

    token, expires_in = (
        create_access_token(
            user_id=int(
                user["id"]
            ),
            username=str(
                user["username"]
            ),
            role=str(
                user["role"]
            ),
        )
    )

    public_user = {
        key: user[key]
        for key in (
            "id",
            "username",
            "role",
            "is_active",
            "created_at",
            "updated_at",
        )
    }

    return {
        "access_token":
            token,
        "token_type":
            "bearer",
        "expires_in":
            expires_in,
        "user":
            public_user,
    }


@router.get(
    "/me",
    response_model=UserResponse,
    summary="获取当前登录用户",
)
def me(
    current_user:
        dict[str, Any]
        = Depends(
            get_current_user
        ),
):
    return current_user
