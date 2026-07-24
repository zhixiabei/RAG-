from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


SESSION_COOKIE_NAME = "rag_session"


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def credentials_match(username: str, password: str, expected_username: str, expected_password: str) -> bool:
    return hmac.compare_digest(username, expected_username) and hmac.compare_digest(password, expected_password)


def create_session(username: str, owner_id: str, secret: str, ttl_seconds: int) -> str:
    payload = {
        "sub": username,
        "owner_id": owner_id,
        "exp": int(time.time()) + ttl_seconds,
    }
    encoded_payload = _encode(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded_payload}.{_encode(signature)}"


def verify_session(token: str | None, secret: str) -> dict[str, Any] | None:
    if not token or token.count(".") != 1:
        return None
    encoded_payload, encoded_signature = token.split(".", 1)
    expected_signature = hmac.new(secret.encode("utf-8"), encoded_payload.encode("ascii"), hashlib.sha256).digest()
    try:
        actual_signature = _decode(encoded_signature)
        payload = json.loads(_decode(encoded_payload).decode("utf-8"))
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not hmac.compare_digest(actual_signature, expected_signature):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("sub"), str) or not isinstance(payload.get("owner_id"), str):
        return None
    try:
        if int(payload.get("exp", 0)) <= int(time.time()):
            return None
    except (TypeError, ValueError):
        return None
    return payload
