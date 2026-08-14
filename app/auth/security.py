"""JWT 与密码安全工具。"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash


PASSWORD_HASH = PasswordHash.recommended()

JWT_ALGORITHM = "HS256"


def _jwt_secret_key() -> str:
    secret = str(
        os.getenv(
            "JWT_SECRET_KEY",
            "",
        )
    ).strip()

    if len(secret) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY 未配置或长度不足 32 个字符"
        )

    return secret


def access_token_expire_minutes() -> int:
    raw = str(
        os.getenv(
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
            "60",
        )
    ).strip()

    try:
        minutes = int(raw)
    except ValueError as error:
        raise RuntimeError(
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES 必须是整数"
        ) from error

    if not 5 <= minutes <= 10080:
        raise RuntimeError(
            "JWT 有效期必须在 5～10080 分钟之间"
        )

    return minutes


def hash_password(password: str) -> str:
    return PASSWORD_HASH.hash(
        str(password)
    )


def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    try:
        return bool(
            PASSWORD_HASH.verify(
                str(password),
                str(password_hash),
            )
        )
    except Exception:
        return False


def create_access_token(
    *,
    user_id: int,
    username: str,
    role: str,
) -> tuple[str, int]:
    now = datetime.now(
        timezone.utc
    )

    minutes = (
        access_token_expire_minutes()
    )

    expires_at = (
        now
        + timedelta(
            minutes=minutes
        )
    )

    payload = {
        "sub": str(user_id),
        "username": str(username),
        "role": str(role),
        "type": "access",
        "iat": now,
        "exp": expires_at,
    }

    token = jwt.encode(
        payload,
        _jwt_secret_key(),
        algorithm=JWT_ALGORITHM,
    )

    return (
        token,
        minutes * 60,
    )


def decode_access_token(
    token: str,
) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            str(token),
            _jwt_secret_key(),
            algorithms=[
                JWT_ALGORITHM
            ],
            options={
                "require": [
                    "sub",
                    "exp",
                    "iat",
                    "type",
                ]
            },
        )

    except InvalidTokenError as error:
        raise ValueError(
            "访问令牌无效或已经过期"
        ) from error

    if payload.get("type") != "access":
        raise ValueError(
            "访问令牌类型错误"
        )

    try:
        payload["user_id"] = int(
            payload["sub"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "访问令牌缺少合法用户 ID"
        ) from error

    return payload
