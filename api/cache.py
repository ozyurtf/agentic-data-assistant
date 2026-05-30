import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
import redis
from dotenv import load_dotenv
from pydantic import BaseModel
from models import RedisConfig, FileMetadata

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
logger = logging.getLogger(__name__)

redis_config = RedisConfig(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True,
    password=os.getenv("REDIS_PASSWORD"),
)

redis_kwargs = redis_config.dict()
if not redis_kwargs.get("password"):
    redis_kwargs.pop("password", None)
r = redis.Redis(**redis_kwargs)

def clear_user_cache(user_id: str) -> int:
    """Clear all cached data for a specific user when a new file is uploaded"""
    try:
        all_user_keys = r.keys(f"*:{user_id}:*")
        logger.info(f"Found {len(all_user_keys)} total keys for user {user_id}: {all_user_keys}")

        patterns = [
            f"cache:{user_id}:*",
            f"schema:{user_id}:*",
        ]

        total_cleared = 0
        for pattern in patterns:
            keys = r.keys(pattern)
            if keys:
                r.delete(*keys)
                total_cleared += len(keys)
                logger.info(f"Cleared {len(keys)} keys with pattern {pattern}")

        logger.info(f"Total cleared: {total_cleared} cache entries for user: {user_id}")
        return total_cleared

    except redis.RedisError as e:
        logger.error(f"Failed to clear cache for user {user_id}: {e}")
        return 0

def push_file_to_stack(user_id: str, file_metadata: FileMetadata) -> bool:
    """Push file metadata to Redis list"""
    try:
        r.lpush(f"files:{user_id}", file_metadata.json())
        return True
    except redis.RedisError as e:
        logger.error(f"Failed to push file data to Redis: {e}")
        return False

def get_latest_file(user_id: str) -> Optional[FileMetadata]:
    """Get most recent uploaded file metadata"""
    try:
        raw = r.lindex(f"files:{user_id}", 0)
        if raw:
            return FileMetadata.parse_raw(raw)
        return None
    except (redis.RedisError, ValueError) as e:
        logger.error(f"Failed to get latest file for user {user_id}: {e}")
        return None

def pop_latest_file(user_id: str) -> Optional[FileMetadata]:
    """Remove most recent uploaded file"""
    try:
        raw = r.lpop(f"files:{user_id}")
        if raw:
            return FileMetadata.parse_raw(raw)
        return None
    except (redis.RedisError, ValueError) as e:
        logger.error(f"Failed to pop latest file for user {user_id}: {e}")
        return None

def safe_cache_get(key: str) -> Optional[dict]:
    """Safely get data from cache with error handling"""
    try:
        cached = r.get(key)
        return json.loads(cached) if cached else None
    except (redis.RedisError, json.JSONDecodeError) as e:
        logger.error(f"Failed to get cache key {key}: {e}")
        return None

def safe_cache_set(key: str, data: Any, ex: int = 3600) -> bool:
    """Safely set data in cache with error handling"""
    try:
        if isinstance(data, BaseModel):
            r.set(key, data.json(), ex=ex)
        else:
            r.set(key, json.dumps(data), ex=ex)
        return True
    except (redis.RedisError, TypeError, ValueError) as e:
        logger.error(f"Failed to set cache key {key}: {e}")
        return False