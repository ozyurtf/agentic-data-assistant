import json
import os
import time
from pathlib import Path
from typing import Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

ph = PasswordHasher()


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


JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-change-in-prod-please")
JWT_ALG = "HS256"
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", 60 * 60 * 24 * 7))


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


USER_STORE_PATH = Path(
    os.getenv(
        "USER_STORE_PATH",
        Path(__file__).resolve().parent.parent / "chatbot" / "user.json",
    )
)


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
