from pathlib import Path
from models import *
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Body, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pymavlink import mavutil
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import redis
import json
from collections import defaultdict
import math
import logging
from typing import List, Dict, Any, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_user_id(request: Request):
    return request.headers.get("user-id", request.client.host)

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"), 
    port=int(os.getenv("REDIS_PORT", 6379)), 
    db=0, 
    decode_responses=True,
)

upload_dir = Path("files")
upload_dir.mkdir(exist_ok=True)

app = FastAPI(
    title="Drone Log API", 
    description="API for processing drone flight logs", 
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"]
)

limiter = Limiter(key_func=get_user_id)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

def clear_user_cache(user_id: str):
    """Clear all cached data for a specific user when a new file is uploaded"""
    try:
        # First, let's see what keys exist for this user
        all_user_keys = r.keys(f"*:{user_id}:*")
        logger.info(f"Found {len(all_user_keys)} total keys for user {user_id}: {all_user_keys}")
        
        # Clear different types of cached data
        patterns = [
            f"cache:{user_id}:*",      # Data cache
            f"schema:{user_id}:*",     # Schema cache
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

def push_file_to_stack(user_id: str, file_data: dict) -> bool:
    """Push file metadata to Redis list"""
    try:
        r.lpush(f"files:{user_id}", json.dumps(file_data))
        return True
    except redis.RedisError as e:
        logger.error(f"Failed to push file data to Redis: {e}")
        return False

def get_latest_file(user_id: str) -> Optional[dict]:
    """Get most recent uploaded file metadata"""
    try:
        raw = r.lindex(f"files:{user_id}", 0)
        return json.loads(raw) if raw else None
    except (redis.RedisError, json.JSONDecodeError) as e:
        logger.error(f"Failed to get latest file for user {user_id}: {e}")
        return None

def pop_latest_file(user_id: str) -> Optional[dict]:
    """Remove most recent uploaded file"""
    try:
        raw = r.lpop(f"files:{user_id}")
        return json.loads(raw) if raw else None
    except (redis.RedisError, json.JSONDecodeError) as e:
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
        r.set(key, json.dumps(data), ex=ex)
        return True
    except (redis.RedisError, TypeError, ValueError) as e:
        logger.error(f"Failed to set cache key {key}: {e}")
        return False

def clean_and_remove_empty_columns(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Efficiently remove columns that are entirely None/NaN.
    Single pass to identify valid columns, single pass to clean.
    """
    if not data_list:
        return data_list
    
    # First pass: identify columns with at least one valid value
    valid_columns = set()
    columns_to_check = set(data_list[0].keys())  # Start with first row's columns
    
    for row in data_list:
        # Check remaining columns that haven't been validated yet
        cols_to_remove = set()
        for col in columns_to_check:
            if col in row:
                value = row[col]
                # Found a valid value for this column
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    valid_columns.add(col)
                    cols_to_remove.add(col)
        
        # Remove validated columns from future checks
        columns_to_check -= cols_to_remove
        
        # Early exit if all columns are validated
        if not columns_to_check:
            break
    
    # Second pass: build cleaned data with only valid columns and clean NaN values
    cleaned_data = []
    for row in data_list:
        cleaned_row = {}
        for col in valid_columns:
            if col in row:
                value = row[col]
                # Replace NaN with None for JSON compatibility
                if isinstance(value, float) and math.isnan(value):
                    cleaned_row[col] = None
                else:
                    cleaned_row[col] = value
        cleaned_data.append(cleaned_row)
    
    return cleaned_data    

def validate_col_map(col_map: Dict[str, List[str]]) -> bool:
    """Validate that col_map has the expected structure"""
    if not isinstance(col_map, dict):
        return False
    
    for msg_type, fields in col_map.items():
        if not isinstance(msg_type, str) or not isinstance(fields, list):
            return False
        if not all(isinstance(field, str) for field in fields):
            return False
    return True

@app.get("/api/current-user", description="Get the current user ID from the chatbot session")
async def get_current_user(request: Request):
    """Get the current user ID from the chatbot session"""
    try:
        # Try to get user ID from headers first (for direct API calls)
        user_id = request.headers.get("user-id")
        if user_id:
            return {"current_user": user_id}
        
        # If no user-id header, try to get from session or return default
        # For now, return a default since we can't access Chainlit session directly
        return {"current_user": "anonymous", "note": "No user-id header provided"}
        
    except Exception as e:
        logger.error(f"Failed to get current user: {str(e)}")
        return {"current_user": "anonymous", "error": "Failed to get current user"}

@app.post("/api/files/{file_id}", response_model=FileReceiveResponse, status_code=201, description="Upload a drone flight log file")
@limiter.limit("10/hour")
async def receive_file(request: Request, file_id: str, file: UploadFile = File(...), user_id: str = Header(alias="user-id")):
    logger.info(f"Receiving file upload: {file_id} for user: {user_id}")
    logger.info(f"File name: {file.filename}")

    if not file.filename.endswith(('.bin', '.log', '.tlog')):
        raise HTTPException(status_code=400, detail="Only .bin, .log, and .tlog files are supported")
    
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size is 100MB")

    file_name = file.filename
    file_path = upload_dir / f"{file_id}_{file_name}"

    try:
        # Stream instead of loading whole file into memory
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)

        file_data = {
            "file_id": file_id,
            "file_path": str(file_path),
            "filename": file.filename
        }

        # Clear cache and store file data (continue even if Redis fails)
        clear_user_cache(user_id)
        push_file_to_stack(user_id, file_data)

        return FileReceiveResponse(**file_data)

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.get("/api/files/{file_id}/schema", description="Get message types and fields from log file")
@limiter.limit("20/hour")
async def get_file_schema(request: Request, file_id: str, user_id: str = Header(alias="user-id")):
    file_data = get_latest_file(user_id)
    if not file_data:
        raise HTTPException(status_code=404, detail="No file found for this user")
    
    # Check cache first (gracefully handle cache failures)
    cache_key = f"schema:{user_id}:{file_data['file_id']}"
    cached_schema = safe_cache_get(cache_key)
    if cached_schema:
        logger.info(f"Schema served from cache for {cache_key}")
        return cached_schema
    
    try:
        # Parse file for schema
        mlog = mavutil.mavlink_connection(file_data["file_path"])
        msg_info = defaultdict(set)
        
        while True:
            msg = mlog.recv_match()
            if msg is None:
                break
            msg_type = msg.get_type()
            msg_info[msg_type].update(msg.to_dict().keys())
        
        schema = {k: sorted(v) for k, v in msg_info.items()}
        result = {"schema": schema, "file_id": file_data["file_id"]}
        
        # Try to cache the schema (continue even if caching fails)
        safe_cache_set(cache_key, result, ex=7200)
        
        return result
        
    except Exception as e:
        logger.error(f"Schema generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate schema: {str(e)}")

@app.post("/api/process", description="Process a drone flight log file")
@limiter.limit("30/hour")
async def process_file(request: Request, col_map: Dict[str, List[str]] = Body(...), user_id: str = Header(alias="user-id")):
    # Validate input
    if not validate_col_map(col_map):
        return JSONResponse(
            status_code=400, 
            content={"success": False, "error": "Invalid col_map format", "data": None}
        )

    file_data = get_latest_file(user_id)
    if not file_data:
        return JSONResponse(
            status_code=404, 
            content={"success": False, "error": "No file found for this user", "data": None}
        )

    try:
        data = {}
        rem_msg_types = []
        
        # Check cache for each message type
        for msg_type in col_map.keys():
            cache_key = f"cache:{user_id}:{file_data['file_id']}:{msg_type}:ALL_FIELDS"
            cached = safe_cache_get(cache_key)
            
            if cached:
                data[msg_type] = cached
                logger.info(f"--------------------------------")
                logger.info(f"Full data received from cache for {msg_type}.")
                logger.info(f"--------------------------------")
            else:
                rem_msg_types.append(msg_type)

        # Process uncached message types
        if rem_msg_types:
            try:
                mlog = mavutil.mavlink_connection(file_data["file_path"])
                
                # Initialize with empty lists
                for msg in rem_msg_types:
                    data[msg] = []
                
                # Read all messages
                msg_count = 0
                while True:
                    msg = mlog.recv_match(type=rem_msg_types)
                    if msg is None:
                        break
                    
                    msg_count += 1
                    msg_type = msg.get_type()
                    if msg_type in col_map:
                        data[msg_type].append(msg.to_dict())
                
                logger.info(f"Processed {msg_count} total messages")
                
            except Exception as e:
                logger.error(f"MAVLink processing failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to process MAVLink data: {str(e)}")
            
            # Clean and cache each message type
            for msg_type in rem_msg_types:
                if data[msg_type]:
                    # Clean in one efficient operation
                    original_count = len(data[msg_type])
                    original_cols = len(data[msg_type][0]) if data[msg_type] else 0
                    
                    data[msg_type] = clean_and_remove_empty_columns(data[msg_type])
                    
                    final_cols = len(data[msg_type][0]) if data[msg_type] else 0
                    logger.info(f"  {msg_type}: {original_count} rows, {original_cols} → {final_cols} columns")
                
                # Try to cache the cleaned data (continue if caching fails)
                cache_key = f"cache:{user_id}:{file_data['file_id']}:{msg_type}:ALL_FIELDS"
                safe_cache_set(cache_key, data[msg_type], ex=3600)

        return JSONResponse(status_code=200, content={
            "success": True,
            "message": "File processed successfully",
            "data": data,
            "metadata": {"file_id": file_data["file_id"], "filename": file_data["filename"]}
        })

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error in process_file: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": f"Failed to process file: {str(e)}",
            "data": None,
            "metadata": {"file_id": file_data.get("file_id", "unknown")}
        })

@app.get("/api/files", description="Get the most recent uploaded file for the user")
@limiter.limit("100/minute")
async def get_file(request: Request, user_id: str = Header(alias="user-id")):
    file_data = get_latest_file(user_id)
    if not file_data:
        raise HTTPException(status_code=404, detail="No files available for this user")
    return file_data

@app.delete("/api/files", description="Delete the most recent uploaded file for the user")
@limiter.limit("5/minute")
async def delete_file(request: Request, user_id: str = Header(...)):
    file_data = pop_latest_file(user_id)
    if not file_data:
        raise HTTPException(status_code=404, detail="No files available for this user")

    file_path = Path(file_data['file_path'])
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
        except OSError as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            # Continue anyway - file metadata is removed from Redis

    return {"message": f"Top file '{file_data['file_id']}' deleted successfully"}

@app.get("/api/debug/redis-keys/{user_id}", description="Debug endpoint to see Redis keys for a user")
async def debug_redis_keys(user_id: str):
    """Debug endpoint to inspect Redis keys for a user"""
    try:
        all_keys = r.keys("*")
        user_keys = r.keys(f"*{user_id}*")
        patterns = [
            f"cache:{user_id}:*",
            f"schema:{user_id}:*", 
            f"files:{user_id}",
        ]
        
        pattern_results = {}
        for pattern in patterns:
            pattern_results[pattern] = r.keys(pattern)
            
        return {
            "user_id": user_id,
            "total_keys": len(all_keys),
            "user_keys": user_keys,
            "pattern_results": pattern_results,
            "sample_keys": all_keys[:10]  # Show first 10 keys as sample
        }
    except redis.RedisError as e:
        return {"error": f"Redis error: {e}"}

@app.get("/health")
async def health_check():
    # Check Redis connectivity
    redis_status = "healthy"
    try:
        r.ping()
    except redis.RedisError:
        redis_status = "unhealthy"
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "upload_dir": str(upload_dir.absolute())
    }