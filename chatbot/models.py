from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import os

class WebContentResult(BaseModel):
    """Result from web content loading"""
    web_content: str = Field(..., description="Extracted web content in markdown format")

class DataExtractionResult(BaseModel):
    """Result from data extraction process"""
    data: Dict[str, Any] = Field(..., description="Extracted data organized by message type")
    
class AnalysisResult(BaseModel):
    """Base model for analysis results"""
    message: str = Field(..., description="Human readable analysis result")

class AverageResult(AnalysisResult):
    """Result from average calculation"""
    average: str = Field(..., description="Formatted average calculation results")

class SumResult(AnalysisResult):
    """Result from sum calculation"""
    sum: str = Field(..., description="Formatted sum calculation results")

class MinMaxResult(AnalysisResult):
    """Result from min/max analysis"""
    pass

class MaximumResult(MinMaxResult):
    """Result from maximum value analysis"""
    maximum: str = Field(..., description="Maximum values with context and timestamps")

class MinimumResult(MinMaxResult):
    """Result from minimum value analysis"""
    minimum: str = Field(..., description="Minimum values with context and timestamps")

class OscillationResult(AnalysisResult):
    """Result from oscillation detection"""
    oscillations: str = Field(..., description="Oscillation analysis with patterns and scores")

class SuddenChangesResult(AnalysisResult):
    """Result from sudden changes detection"""
    sudden_changes: str = Field(..., description="Detected sudden changes with context")

class OutlierResult(AnalysisResult):
    """Result from outlier detection"""
    outliers: str = Field(..., description="Statistical outlier analysis with multiple methods")

class EventDetectionConfig(BaseModel):
    """Configuration for event detection"""
    message_type: str = Field(..., description="Target message type to analyze")
    condition_type: str = Field(..., description="Type of condition: threshold, signal_loss, availability, state_change")
    field: str = Field(..., description="Field to analyze")
    description: str = Field(..., description="Human readable description of the event")
    
    # Optional fields based on condition_type
    operator: Optional[str] = Field(None, description="Comparison operator for threshold conditions")
    value: Optional[Union[int, float]] = Field(None, description="Threshold value")
    loss_indicators: Optional[List[Union[str, int]]] = Field(None, description="List of values indicating signal loss")
    target_state: Optional[str] = Field(None, description="Target state for state_change conditions")
    
    @validator('condition_type')
    def validate_condition_type(cls, v):
        valid_types = ['threshold', 'signal_loss', 'availability', 'state_change']
        if v not in valid_types:
            raise ValueError(f'condition_type must be one of: {valid_types}')
        return v
    
    @validator('operator')
    def validate_operator(cls, v, values):
        if values.get('condition_type') == 'threshold' and v is None:
            raise ValueError('operator is required for threshold conditions')
        if v and v not in ['>=', '<=', '>', '<', '==', '!=']:
            raise ValueError('operator must be one of: >=, <=, >, <, ==, !=')
        return v

class EventOccurrence(BaseModel):
    """Single event occurrence"""
    message_type: str
    event_description: str
    field_checked: str
    event_value: Any
    full_context: Dict[str, Any]
    timestamp: Optional[Any] = None

class EventResult(AnalysisResult):
    """Result from event detection"""
    events: str = Field(..., description="Formatted event detection results")
    occurrences: Optional[List[EventOccurrence]] = Field(None, description="Structured event occurrences")

class VisualizationResult(BaseModel):
    """Result from visualization generation"""
    message: str = Field(..., description="Status message about visualization")
    code_generated: bool = Field(False, description="Whether visualization code was generated")

# Session data models
class UserSession(BaseModel):
    """User session data structure"""
    msg_context: str = Field(default="", description="Message context from log file schema")
    file_id: str = Field(default="", description="Current file ID")
    web_content: str = Field(default="", description="Cached web content")
    data: Dict[str, Any] = Field(default_factory=dict, description="Extracted log data")
    col_map: Dict[str, List[str]] = Field(default_factory=dict, description="Column mapping for data extraction")
    code: str = Field(default="", description="Generated visualization code")
    message_history: List[Any] = Field(default_factory=list, description="Chat message history")


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
        
        if len(v) > int(os.getenv("MAX_MESSAGE_TYPES", 3)):
            raise ValueError(f'Maximum {os.getenv("MAX_MESSAGE_TYPES", 3)} message types allowed per request')
        
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

# API interaction models
class FileInfo(BaseModel):
    """File information from API"""
    file_path: str
    file_id: str

class SchemaData(BaseModel):
    """Schema data from API"""
    schema: Dict[str, List[str]]

class ProcessResponse(BaseModel):
    """Response from process API endpoint"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Tool result union type
ToolResult = Union[
    WebContentResult,
    DataExtractionResult,
    AverageResult,
    SumResult,
    MaximumResult,
    MinimumResult,
    OscillationResult,
    SuddenChangesResult,
    OutlierResult,
    EventResult,
    VisualizationResult,
    str  # For error messages
]