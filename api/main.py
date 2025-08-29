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
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_user_id(request: Request):
    return request.headers.get("user-id", request.client.host)

# Initialize Redis with configuration from models
redis_config = RedisConfig(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6380)),
    db=0,
    decode_responses=True
)

r = redis.Redis(**redis_config.dict())

# Initialize app configuration
app_config = AppConfig(
    upload_dir=Path("files"),
    max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", 100)),
    cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", 3600)),
    max_message_types_per_request=int(os.getenv("MAX_MESSAGE_TYPES", 3)),
    allowed_file_extensions=['.bin', '.log', '.tlog']
)

app_config.upload_dir.mkdir(exist_ok=True)

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

def clear_user_cache(user_id: str) -> int:
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

@app.get("/api/current-user", response_model=UserResponse, description="Get the current user ID from the chatbot session")
async def get_current_user(request: Request):
    """Get the current user ID from the chatbot session"""
    try:
        # Try to get user ID from headers first (for direct API calls)
        user_id = request.headers.get("user-id")
        if user_id:
            return UserResponse(current_user=user_id)
        
        # If no user-id header, try to get from session or return default
        # For now, return a default since we can't access Chainlit session directly
        return UserResponse(
            current_user="anonymous", 
            note="No user-id header provided"
        )
        
    except Exception as e:
        logger.error(f"Failed to get current user: {str(e)}")
        return UserResponse(
            current_user="anonymous", 
            error="Failed to get current user"
        )

@app.post("/api/files/{file_id}", response_model=FileReceiveResponse, status_code=201, description="Upload a drone flight log file")
@limiter.limit("10/hour")
async def receive_file(request: Request, file_id: str, file: UploadFile = File(...), user_id: str = Header(alias="user-id")):
    logger.info(f"Receiving file upload: {file_id} for user: {user_id}")
    logger.info(f"File name: {file.filename}")

    # Validate file extension
    if not any(file.filename.lower().endswith(ext) for ext in app_config.allowed_file_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Only {', '.join(app_config.allowed_file_extensions)} files are supported"
        )
    
    # Validate file size
    if file.size and file.size > app_config.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Max size is {app_config.max_file_size_mb}MB"
        )

    file_name = file.filename
    file_path = app_config.upload_dir / f"{file_id}_{file_name}"

    try:
        # Stream instead of loading whole file into memory
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)

        # Create file metadata using the Pydantic model
        file_metadata = FileMetadata(
            file_id=file_id,
            file_path=str(file_path),
            filename=file.filename,
            file_size=file_path.stat().st_size if file_path.exists() else None
        )

        # Clear cache and store file data (continue even if Redis fails)
        clear_user_cache(user_id)
        push_file_to_stack(user_id, file_metadata)

        return FileReceiveResponse(
            file_id=file_metadata.file_id,
            file_path=file_metadata.file_path,
            filename=file_metadata.filename
        )

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        logger.error(f"File upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.get("/api/files/{file_id}/schema", response_model=SchemaResponse, description="Get message types and fields from log file")
@limiter.limit("20/hour")
async def get_file_schema(request: Request, file_id: str, user_id: str = Header(alias="user-id")):
    file_metadata = get_latest_file(user_id)
    if not file_metadata:
        raise HTTPException(status_code=404, detail="No file found for this user")
    
    # Check cache first (gracefully handle cache failures)
    cache_key = f"schema:{user_id}:{file_metadata.file_id}"
    cached_schema = safe_cache_get(cache_key)
    if cached_schema:
        logger.info(f"Schema served from cache for {cache_key}")
        return SchemaResponse(**cached_schema)
    
    try:
        # Parse file for schema
        mlog = mavutil.mavlink_connection(file_metadata.file_path)
        msg_info = defaultdict(set)
        
        while True:
            msg = mlog.recv_match()
            if msg is None:
                break
            msg_type = msg.get_type()
            msg_info[msg_type].update(msg.to_dict().keys())
        
        schema = {k: sorted(v) for k, v in msg_info.items()}
        
        schema_response = SchemaResponse(
            schema=schema,
            file_id=file_metadata.file_id,
            total_message_types=len(schema)
        )
        
        # Try to cache the schema (continue even if caching fails)
        safe_cache_set(cache_key, schema_response, ex=7200)
        
        return schema_response
        
    except Exception as e:
        logger.error(f"Schema generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate schema: {str(e)}")

@app.post("/api/process", description="Process a drone flight log file")
@limiter.limit("30/hour")
async def process_file(
    request: Request, 
    process_request: ColMapRequest,
    user_id: str = Header(alias="user-id")
):
    """Process drone flight log file with column mapping validation"""
    
    # Extract validated col_map from the request object
    validated_col_map = process_request.col_map  # Changed: Extract from request object
    
    file_metadata = get_latest_file(user_id)
    if not file_metadata:
        return ProcessErrorResponse(
            error="No file found for this user"
        )

    try:
        data = {}
        rem_msg_types = []
        cache_hits = 0
        cache_misses = 0
        
        # Check cache for each message type
        for msg_type in validated_col_map.keys():
            cache_key = f"cache:{user_id}:{file_metadata.file_id}:{msg_type}:ALL_FIELDS"
            cached = safe_cache_get(cache_key)
            
            if cached:
                data[msg_type] = cached
                cache_hits += 1
                logger.info(f"Full data received from cache for {msg_type}.")
            else:
                rem_msg_types.append(msg_type)
                cache_misses += 1

        # Process uncached message types
        total_messages_read = 0
        processing_start = datetime.utcnow()
        
        if rem_msg_types:
            try:
                mlog = mavutil.mavlink_connection(file_metadata.file_path)
                
                # Initialize with empty lists
                for msg in rem_msg_types:
                    data[msg] = []
                
                # Read all messages
                while True:
                    msg = mlog.recv_match(type=rem_msg_types)
                    if msg is None:
                        break
                    
                    total_messages_read += 1
                    msg_type = msg.get_type()
                    if msg_type in validated_col_map:
                        data[msg_type].append(msg.to_dict())
                
                logger.info(f"Processed {total_messages_read} total messages")
                
            except Exception as e:
                logger.error(f"MAVLink processing failed: {str(e)}")
                return ProcessErrorResponse(
                    error=f"Failed to process MAVLink data: {str(e)}"
                )
            
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
                cache_key = f"cache:{user_id}:{file_metadata.file_id}:{msg_type}:ALL_FIELDS"
                safe_cache_set(cache_key, data[msg_type], ex=app_config.cache_ttl_seconds)

        processing_time = (datetime.utcnow() - processing_start).total_seconds()
        
        # Create processing statistics
        processing_stats = ProcessingStats(
            total_messages_read=total_messages_read,
            message_types_found=len([msg_type for msg_type in data.keys() if data[msg_type]]),
            processing_time_seconds=processing_time,
            cache_hits=cache_hits,
            cache_misses=cache_misses
        )

        return ProcessSuccessResponse(
            message="File processed successfully",
            data=data,
            metadata={
                "file_id": file_metadata.file_id, 
                "filename": file_metadata.filename,
                "processing_stats": processing_stats.dict()
            }
        )

    except ValueError as e:
        # This will catch Pydantic validation errors
        logger.error(f"Validation error: {str(e)}")
        return ProcessErrorResponse(
            error=f"Validation error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in process_file: {str(e)}", exc_info=True)
        return ProcessErrorResponse(
            error=f"Failed to process file: {str(e)}",
            metadata={"file_id": file_metadata.file_id if file_metadata else "unknown"}
        )

@app.get("/api/files", description="Get the most recent uploaded file for the user")
@limiter.limit("100/minute")
async def get_file(request: Request, user_id: str = Header(alias="user-id")):
    file_metadata = get_latest_file(user_id)
    if not file_metadata:
        raise HTTPException(status_code=404, detail="No files available for this user")
    return file_metadata

@app.delete("/api/files", response_model=DeleteResponse, description="Delete the most recent uploaded file for the user")
@limiter.limit("5/minute")
async def delete_file(request: Request, user_id: str = Header(alias="user-id")):
    file_metadata = pop_latest_file(user_id)
    if not file_metadata:
        raise HTTPException(status_code=404, detail="No files available for this user")

    file_path = Path(file_metadata.file_path)
    if file_path.exists():
        try:
            file_path.unlink()
            logger.info(f"Deleted file: {file_path}")
        except OSError as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            # Continue anyway - file metadata is removed from Redis

    return DeleteResponse(
        message=f"File '{file_metadata.file_id}' deleted successfully",
        deleted_file_id=file_metadata.file_id
    )

@app.get("/api/debug/redis-keys/{user_id}", response_model=RedisDebugResponse, description="Debug endpoint to see Redis keys for a user")
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
            
        return RedisDebugResponse(
            user_id=user_id,
            total_keys=len(all_keys),
            user_keys=user_keys,
            pattern_results=pattern_results,
            sample_keys=all_keys[:10]  # Show first 10 keys as sample
        )
    except redis.RedisError as e:
        return RedisDebugResponse(
            user_id=user_id,
            total_keys=0,
            user_keys=[],
            pattern_results={},
            sample_keys=[],
            error=f"Redis error: {e}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Check Redis connectivity
    redis_status = "healthy"
    try:
        r.ping()
    except redis.RedisError:
        redis_status = "unhealthy"
    
    return HealthResponse(
        status="healthy" if redis_status == "healthy" else "degraded",
        redis=redis_status,
        upload_dir=str(app_config.upload_dir.absolute())
    )