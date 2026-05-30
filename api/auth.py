import json
import os
import time
from pathlib import Path
from typing import Optional
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from dotenv import load_dotenv
from fastapi import Request, Response

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
ph = PasswordHasher()

JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-change-in-prod-please")
JWT_ALG = "HS256"
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", 60 * 60 * 24 * 7))

USER_STORE_PATH = Path(
    os.getenv(
        "USER_STORE_PATH",
        Path(__file__).resolve().parent.parent / "chatbot" / "user.json",
    )
)

COOKIE_NAME = "auth_token"
COOKIE_DOMAIN = os.getenv("AUTH_COOKIE_DOMAIN")
COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax").lower()
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"

def hash_password(password: str) -> str:
    return ph.hash(password)

def verify_password(stored_hash: str, password: str) -> bool:
    try:
        ph.verify(stored_hash, password)
        return True
    except VerifyMismatchError:
        return False
    except Exception:
        return False

def issue_jwt(user_id: str) -> str:
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + JWT_TTL_SECONDS,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def verify_jwt(token: str) -> Optional[dict]:
    if not token:
        return None
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.PyJWTError:
        return None

def load_users() -> list[dict]:
    if not USER_STORE_PATH.exists() or USER_STORE_PATH.stat().st_size == 0:
        return []
    with open(USER_STORE_PATH, "r") as f:
        return json.load(f)

def save_users(users: list[dict]) -> None:
    USER_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_STORE_PATH, "w") as f:
        json.dump(users, f, indent=2)

def find_user(user_id: str) -> Optional[dict]:
    for u in load_users():
        if u.get("user_id") == user_id:
            return u
    return None

def add_user(user_id: str, password: str) -> dict:
    users = load_users()
    if any(u.get("user_id") == user_id for u in users):
        raise ValueError(f"user '{user_id}' already exists")
    user = {"user_id": user_id, "password_hash": hash_password(password)}
    users.append(user)
    save_users(users)
    return user

def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        samesite=COOKIE_SAMESITE,
        secure=COOKIE_SECURE,
        max_age=JWT_TTL_SECONDS,
        domain=COOKIE_DOMAIN,
        path="/",
    )

def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=COOKIE_NAME,
        domain=COOKIE_DOMAIN,
        path="/",
    )

def current_user_from_request(request: Request, auth_token: Optional[str] = None) -> Optional[str]:
    token = auth_token or request.cookies.get(COOKIE_NAME)
    if not token:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header.split(" ", 1)[1]
    payload = verify_jwt(token) if token else None
    return payload.get("sub") if payload else None