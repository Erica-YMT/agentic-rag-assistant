from app.auth.email_verification import EmailVerificationStore


def test_email_code_is_single_use(monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    store = EmailVerificationStore(ttl_seconds=60)
    code = store.issue("User@Example.com")

    assert store.verify("user@example.com", code)
    assert not store.verify("user@example.com", code)


def test_email_code_expiry(monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    store = EmailVerificationStore(ttl_seconds=60)
    code = store.issue("user@example.com")
    store._values["user@example.com"] = (code, 0)

    assert not store.verify("user@example.com", code)
