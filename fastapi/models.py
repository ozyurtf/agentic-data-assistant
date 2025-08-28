from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Request Models
class ColMapRequest(BaseModel):
    """Request model for processing drone flight logs with column mapping"""
    col_map: Dict[str, List[str]] = Field(
        ..., 
        description="Mapping of MAVLink message types to their field lists",
        example={"GPS": ["Lat", "Lng", "Alt"], "ATT": ["Roll", "Pitch", "Yaw"]}
    )
    
    @validator('col_map')
    def validate_col_map(cls, v):
        if not isinstance(v, dict):
            raise ValueError('col_map must be a dictionary')
        
        if not v:
            raise ValueError('col_map cannot be empty')
        
        if len(v) > 3:
            raise ValueError('Maximum 3 message types allowed per request')
        
        for msg_type, fields in v.items():
            if not isinstance(msg_type, str) or not msg_type.strip():
                raise ValueError(f'Message type must be a non-empty string, got: {type(msg_type).__name__}')
            
            if not isinstance(fields, list):
                raise ValueError(f'Fields for {msg_type} must be a list, got: {type(fields).__name__}')
            
            if not fields:
                raise ValueError(f'Fields list for {msg_type} cannot be empty')
            
            if len(fields) > 50:  # Reasonable limit
                raise ValueError(f'Maximum 50 fields per message type, got {len(fields)} for {msg_type}')
            
            for field in fields:
                if not isinstance(field, str) or not field.strip():
                    raise ValueError(f'All fields must be non-empty strings, got: {field}')
        
        return v

# Data Models
class FileMetadata(BaseModel):
    """Model for file metadata stored in Redis"""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    file_path: str = Field(..., description="Path where the file is stored on disk")
    filename: str = Field(..., description="Original filename as uploaded by user")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Upload timestamp")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        path = Path(v)
        if not path.suffix.lower() in ['.bin', '.log', '.tlog']:
            raise ValueError('File must have .bin, .log, or .tlog extension')
        return str(path)

class CacheEntry(BaseModel):
    """Model for cached data entries in Redis"""
    message_type: str
    data: List[Dict[str, Any]]
    cached_at: datetime = Field(default_factory=datetime.utcnow)
    field_count: int = Field(..., description="Number of fields in the cached data")
    row_count: int = Field(..., description="Number of rows in the cached data")

# Response Models
class FileReceiveResponse(BaseModel):
    """Response model for successful file upload"""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    file_path: str = Field(..., description="Server path where file is stored")
    filename: str = Field(..., description="Original filename")
    message: str = Field(default="File uploaded successfully", description="Success message")

class SchemaResponse(BaseModel):
    """Response model for file schema endpoint"""
    schema: Dict[str, List[str]] = Field(
        ..., 
        description="MAVLink message types and their available fields",
        example={"GPS": ["TimeUS", "Status", "Lat", "Lng"], "ATT": ["TimeUS", "Roll", "Pitch"]}
    )
    file_id: str = Field(..., description="File ID that this schema belongs to")
    total_message_types: int = Field(..., description="Total number of message types found")
    
    @validator('total_message_types', always=True)
    def set_message_type_count(cls, v, values):
        if 'schema' in values:
            return len(values['schema'])
        return v

class ProcessSuccessResponse(BaseModel):
    """Response model for successful file processing"""
    success: bool = Field(True, description="Processing success status")
    message: str = Field(..., description="Success message")
    data: Dict[str, List[Dict[str, Any]]] = Field(
        ..., 
        description="Processed MAVLink data organized by message type"
    )
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the processing")

class ProcessErrorResponse(BaseModel):
    """Response model for failed file processing"""
    success: bool = Field(False, description="Processing success status")
    error: str = Field(..., description="Error message describing what went wrong")
    data: Optional[Dict[str, Any]] = Field(None, description="Partial data if any was processed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata about the failed processing")

class UserResponse(BaseModel):
    """Response model for current user endpoint"""
    current_user: str = Field(..., description="Current user identifier")
    note: Optional[str] = Field(None, description="Additional information about user identification")
    error: Optional[str] = Field(None, description="Error message if user identification failed")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Overall system health status")
    redis: str = Field(..., description="Redis connection status")
    upload_dir: str = Field(..., description="Upload directory path")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")

class DeleteResponse(BaseModel):
    """Response model for file deletion"""
    message: str = Field(..., description="Deletion confirmation message")
    deleted_file_id: str = Field(..., description="ID of the deleted file")
    deleted_at: datetime = Field(default_factory=datetime.utcnow, description="Deletion timestamp")

# Debug Models
class RedisDebugResponse(BaseModel):
    """Debug response showing Redis key information"""
    user_id: str = Field(..., description="User ID being debugged")
    total_keys: int = Field(..., description="Total number of keys in Redis")
    user_keys: List[str] = Field(..., description="All keys related to this user")
    pattern_results: Dict[str, List[str]] = Field(..., description="Keys matching specific patterns")
    sample_keys: List[str] = Field(..., description="Sample of all keys for inspection")
    error: Optional[str] = Field(None, description="Any errors encountered during debug")

# Internal Processing Models
class MAVLinkMessage(BaseModel):
    """Model for individual MAVLink messages"""
    message_type: str = Field(..., description="Type of MAVLink message")
    timestamp: Optional[int] = Field(None, description="Message timestamp in microseconds")
    data: Dict[str, Any] = Field(..., description="Message data fields")

class ProcessingStats(BaseModel):
    """Statistics about data processing"""
    total_messages_read: int = Field(0, description="Total messages read from file")
    message_types_found: int = Field(0, description="Number of different message types found")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to process")
    cache_hits: int = Field(0, description="Number of message types served from cache")
    cache_misses: int = Field(0, description="Number of message types processed from file")
    
# Configuration Models
class RedisConfig(BaseModel):
    """Redis connection configuration"""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    decode_responses: bool = Field(default=True, description="Whether to decode Redis responses")

class AppConfig(BaseModel):
    """Application configuration"""
    upload_dir: Path = Field(default=Path("files"), description="Directory for uploaded files")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    cache_ttl_seconds: int = Field(default=3600, description="Cache time-to-live in seconds")
    max_message_types_per_request: int = Field(default=3, description="Maximum message types per processing request")
    allowed_file_extensions: List[str] = Field(default=['.bin', '.log', '.tlog'], description="Allowed file extensions")

# Error Models
class APIError(BaseModel):
    """Standard API error response"""
    detail: str = Field(..., description="Error description")
    error_code: Optional[str] = Field(None, description="Specific error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(None, description="The invalid value that was provided")