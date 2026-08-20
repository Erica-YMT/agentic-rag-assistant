"""Short-lived email verification codes with optional SMTP delivery."""

from __future__ import annotations

from email.message import EmailMessage
import os
import secrets
import smtplib
import time
from threading import Lock


class EmailDeliveryError(RuntimeError):
    """A delivery failure that is safe to report to an API caller."""


class EmailVerificationStore:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._values: dict[str, tuple[str, float]] = {}
        self._lock = Lock()

    def issue(self, email: str) -> str:
        address = str(email).strip().lower()
        code = f"{secrets.randbelow(1_000_000):06d}"

        # Do not leave a usable code behind when delivery fails.
        self._send(address, code)
        with self._lock:
            self._values[address] = (code, time.time() + self.ttl_seconds)
        return code

    @staticmethod
    def is_delivery_configured() -> bool:
        return bool(str(os.getenv("SMTP_HOST", "")).strip())

    def verify(self, email: str, code: str) -> bool:
        address = str(email).strip().lower()
        with self._lock:
            value = self._values.get(address)
            if value is None or value[1] < time.time() or not secrets.compare_digest(value[0], str(code).strip()):
                return False
            self._values.pop(address, None)
            return True

    @staticmethod
    def _send(address: str, code: str) -> None:
        host = str(os.getenv("SMTP_HOST", "")).strip()
        if not host:
            return
        message = EmailMessage()
        message["Subject"] = "Agentic RAG Assistant verification code"
        message["From"] = os.getenv("SMTP_FROM", "no-reply@example.com")
        message["To"] = address
        message.set_content(f"Your verification code is {code}.")
        try:
            with smtplib.SMTP(
                host,
                int(os.getenv("SMTP_PORT", "587")),
                timeout=10,
            ) as server:
                if os.getenv("SMTP_TLS", "1") == "1":
                    server.starttls()
                username = os.getenv("SMTP_USERNAME")
                password = os.getenv("SMTP_PASSWORD")
                if username and password:
                    server.login(username, password)
                server.send_message(message)
        except (OSError, smtplib.SMTPException) as error:
            raise EmailDeliveryError(
                "SMTP 服务器不可达或拒绝投递。"
            ) from error


email_verification_store = EmailVerificationStore()
