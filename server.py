"""
FastAPI Server for Terminal Chatbot - Unified OpenAI-Compatible API
REST API with PostgreSQL database and S3 storage integration.

Migration: Unified /chat endpoint supporting text, streaming, images, and files
"""

import os
import uuid
import base64
import json
import time
import io
from datetime import datetime
from typing import Optional, List, Union, Literal, Any
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import our modules
from logger import setup_logging, get_logger
from exceptions import (
    ChatbotError,
    ConfigurationError,
    RateLimitExceededError,
    CostLimitExceededError,
    ValidationError,
    APIError as ChatbotAPIError
)
from validators import sanitize_input, validate_conversation_id
from rate_limiter import TokenBucketRateLimiter
from config_validator import validate_config
from api_client import create_openai_client, get_retry_decorator
from health import health_check, liveness_check, readiness_check

# Database
try:
    from database import Database, get_database, POSTGRES_AVAILABLE, DatabaseError
    DATABASE_AVAILABLE = POSTGRES_AVAILABLE
except ImportError:
    DATABASE_AVAILABLE = False
    Database = None
    DatabaseError = Exception

# Storage
try:
    from storage import S3Storage, get_storage, S3_AVAILABLE, S3_ENABLED, StorageError
    STORAGE_AVAILABLE = S3_AVAILABLE
except ImportError:
    STORAGE_AVAILABLE = False
    S3_ENABLED = False
    StorageError = Exception

# OpenAI
from openai import OpenAI, APIConnectionError, RateLimitError as OpenAIRateLimitError, APITimeoutError, APIError

# Configuration
DATABASE_ENABLED = os.getenv("DATABASE_ENABLED", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load config
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent


def load_config() -> dict:
    config_path = BASE_DIR / "config.yaml"
    default_config = {
        "model": "gpt-4o",
        "max_history_items": 100,
        "rate_limit_per_minute": 10,
        "max_input_length": 10000,
        "max_file_size_mb": 20,
        "cost_limit_per_session": 5.0,
        "warn_at_cost": 1.0,
        "pricing": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
        "api_timeout": 30,
        "api_max_retries": 3
    }
    if config_path.exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
            default_config.update(user_config)
    return default_config


def load_prompts() -> dict:
    prompts_path = BASE_DIR / "prompts.yaml"
    default_prompts = {
        "system_prompt": "You are a helpful AI assistant. Be concise and helpful.",
    }
    if prompts_path.exists():
        with open(prompts_path, "r") as f:
            user_prompts = yaml.safe_load(f) or {}
            default_prompts.update(user_prompts)
    return default_prompts


CONFIG = validate_config(load_config())
PROMPTS = load_prompts()

# Setup logging
logging_config = CONFIG.get("logging", {})
logger = setup_logging(
    level=logging_config.get("level", "INFO"),
    log_to_file=logging_config.get("log_to_file", True),
    log_dir=logging_config.get("log_dir", "./logs")
)

# Global instances
db: Optional[Database] = None
storage = None
openai_client: Optional[OpenAI] = None
rate_limiters: dict = {}  # Per-user rate limiters


# =============================================================================
# OpenAI-Compatible Pydantic Models
# =============================================================================

class MessageRole(str, Enum):
    """Message roles for chat completions."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class TextContent(BaseModel):
    """Text content part for multimodal messages."""
    type: Literal["text"] = "text"
    text: str


class ImageUrlDetail(str, Enum):
    """Image detail levels."""
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class ImageUrlContent(BaseModel):
    """Image URL content part for multimodal messages."""
    type: Literal["image_url"] = "image_url"
    image_url: dict = Field(..., description="Object with 'url' and optional 'detail'")
    
    @field_validator('image_url')
    @classmethod
    def validate_image_url(cls, v):
        if not isinstance(v, dict) or 'url' not in v:
            raise ValueError("image_url must contain 'url' field")
        return v


class FileContent(BaseModel):
    """File content part for document references."""
    type: Literal["file"] = "file"
    file: dict = Field(..., description="Object with 'url' and 'name'")


ContentPart = Union[TextContent, ImageUrlContent, FileContent]


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: MessageRole
    content: Union[str, List[ContentPart]]
    name: Optional[str] = None
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Content array cannot be empty")
        return v


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="gpt-4o", description="Model to use")
    messages: List[ChatMessage] = Field(..., min_length=1)
    stream: bool = Field(default=False, description="Enable streaming")
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    presence_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(default=0, ge=-2, le=2)
    user_id: Optional[str] = Field(default=None, description="User identifier")
    conversation_id: Optional[str] = Field(default=None, description="Existing conversation")
    store: bool = Field(default=True, description="Persist conversation")
    
    # TextLLM-specific extensions
    system_prompt: Optional[str] = Field(default=None, description="Override system prompt")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello!"}
                ],
                "stream": False,
                "temperature": 1.0
            }
        }


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChoiceMessage(BaseModel):
    """Message in a choice."""
    role: str
    content: Optional[str] = None


class Choice(BaseModel):
    """Response choice."""
    index: int
    message: ChoiceMessage
    finish_reason: Optional[str] = None  # "stop", "length", "content_filter"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: UsageInfo
    
    # TextLLM extensions
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    cost: Optional[float] = None


class DeltaMessage(BaseModel):
    """Delta message for streaming."""
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    """Choice in a streaming chunk."""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]
    usage: Optional[UsageInfo] = None  # Present in final chunk


class ErrorDetail(BaseModel):
    """OpenAI-compatible error detail."""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""
    error: ErrorDetail


class ConversationCreate(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)
    metadata: Optional[dict] = None


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    created_at: str
    message_count: int


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    tokens_input: int = 0
    tokens_output: int = 0
    cost: float = 0


class FileUploadResponse(BaseModel):
    success: bool
    key: Optional[str] = None
    url: Optional[str] = None
    size: Optional[int] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    checks: dict
    uptime: dict


class UserStatsResponse(BaseModel):
    total_conversations: int
    total_messages: int
    total_tokens_input: int
    total_tokens_output: int
    total_cost: float


class SessionResponse(BaseModel):
    id: str
    user_id: str
    created_at: str
    last_activity: str
    total_tokens: int
    total_cost: float
    is_active: bool


# =============================================================================
# Lifespan (Startup/Shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    global db, storage, openai_client

    logger.info("Starting API server...")

    # Initialize OpenAI client
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set!")
        raise ConfigurationError("OPENAI_API_KEY environment variable is required")

    openai_client = create_openai_client(
        api_key=OPENAI_API_KEY,
        timeout=CONFIG.get("api_timeout", 30)
    )
    logger.info("OpenAI client initialized")

    # Initialize Database
    if DATABASE_AVAILABLE and DATABASE_ENABLED:
        try:
            db = Database.initialize(database_url=DATABASE_URL)
            logger.info("Database connected successfully")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            db = None
    else:
        logger.warning("Database disabled or unavailable, using in-memory storage")

    # Initialize Storage (optional - required for file uploads)
    if S3_ENABLED:
        if not STORAGE_AVAILABLE:
            raise ConfigurationError("S3 storage unavailable. Install boto3: pip install boto3")
        storage = S3Storage.initialize()
        logger.info("S3 storage initialized")
    else:
        logger.warning("S3 storage disabled. File uploads will not be available.")

    logger.info("API server started successfully")

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down API server...")
    if db:
        Database.close()
        logger.info("Database connections closed")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Terminal Chatbot API",
    description="Production-ready chatbot API with OpenAI, PostgreSQL, and S3 integration. "
                "OpenAI-compatible /chat endpoint supporting text, streaming, images, and documents.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Dependencies
# =============================================================================

def get_rate_limiter(user_id: str) -> TokenBucketRateLimiter:
    """Get or create rate limiter for user."""
    if user_id not in rate_limiters:
        rate_limiters[user_id] = TokenBucketRateLimiter(
            requests_per_minute=CONFIG.get("rate_limit_per_minute", 10)
        )
    return rate_limiters[user_id]


def get_db():
    """Dependency to get database instance."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


def get_openai():
    """Dependency to get OpenAI client."""
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized")
    return openai_client


# =============================================================================
# Helper Functions
# =============================================================================

def get_or_create_session(user_id: str) -> Optional[str]:
    """
    Get existing session or create new one for a user.
    Returns session_id or None if database unavailable.
    """
    if not db:
        return None

    session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d')}"

    try:
        existing = db.get_session(session_id)
        if existing:
            return session_id

        db.create_session(session_id, user_id)
        logger.info(f"Created session: {session_id}")
        return session_id
    except Exception as e:
        logger.error(f"Session management failed: {e}")
        return None


def update_session_after_chat(session_id: str, tokens: int, cost: float, increment_conversations: bool = False):
    """Update session statistics after a chat."""
    if not db or not session_id:
        return

    try:
        db.update_session_stats(session_id, tokens, cost, increment_conversations)
    except Exception as e:
        logger.error(f"Failed to update session stats: {e}")


def build_chat_messages(
    messages: List[ChatMessage],
    system_prompt: Optional[str] = None,
    file_content_parts: Optional[List[dict]] = None
) -> List[dict]:
    """
    Convert ChatMessage list to OpenAI Chat Completions API messages format.
    
    Args:
        messages: List of chat messages
        system_prompt: Optional system prompt to prepend
        file_content_parts: Optional file content parts to include in the last user message
        
    Returns:
        List of message dicts with 'role' and 'content' keys for Chat Completions API
    """
    chat_messages = []
    
    # Add system message if provided
    if system_prompt:
        chat_messages.append({"role": "system", "content": system_prompt})
    
    # Process each message
    for i, msg in enumerate(messages):
        is_last_message = (i == len(messages) - 1)
        
        if isinstance(msg.content, str):
            # Simple text message
            content = msg.content
            
            # If this is the last user message and we have file content parts, combine them
            if is_last_message and file_content_parts and msg.role.value == "user":
                # Build multimodal content array
                content_parts = [{"type": "text", "text": msg.content}]
                for part in file_content_parts:
                    if part.get("type") == "input_image":
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": part.get("image_url", "")}
                        })
                    elif part.get("type") == "input_file":
                        # For PDFs and other files, include as text reference
                        content_parts.append({
                            "type": "text",
                            "text": f"[File: {part.get('filename', 'document')}]"
                        })
                    elif part.get("type") == "input_text":
                        content_parts.append({"type": "text", "text": part.get("text", "")})
                content = content_parts
            
            chat_messages.append({"role": msg.role.value, "content": content})
        else:
            # Multimodal content (list of ContentPart)
            content_parts = []
            for part in msg.content:
                if part.type == "text":
                    content_parts.append({"type": "text", "text": part.text})
                elif part.type == "image_url":
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": part.image_url.get("url", "")}
                    })
                elif part.type == "file":
                    content_parts.append({
                        "type": "text",
                        "text": f"[File: {part.file.get('name', 'document')}]"
                    })
            
            # Add file content parts to last user message
            if is_last_message and file_content_parts and msg.role.value == "user":
                for part in file_content_parts:
                    if part.get("type") == "input_image":
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": part.get("image_url", "")}
                        })
                    elif part.get("type") == "input_text":
                        content_parts.append({"type": "text", "text": part.get("text", "")})
            
            if content_parts:
                chat_messages.append({"role": msg.role.value, "content": content_parts})
    
    return chat_messages


def extract_text_from_content(messages: List[ChatMessage]) -> str:
    """Extract text content from messages for sanitization."""
    texts = []
    for msg in messages:
        if isinstance(msg.content, str):
            texts.append(msg.content)
        else:
            for part in msg.content:
                if part.type == "text":
                    texts.append(part.text)
    return " ".join(texts)


def calculate_cost(usage: dict) -> float:
    """Calculate cost from usage."""
    pricing = CONFIG.get("pricing", {"input_per_1k": 0.0025, "output_per_1k": 0.01})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return round(
        (input_tokens / 1000) * pricing["input_per_1k"] +
        (output_tokens / 1000) * pricing["output_per_1k"],
        6
    )


def extract_usage(response) -> dict:
    """Extract usage from OpenAI Chat Completions API response."""
    if not response.usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Chat Completions API uses prompt_tokens, completion_tokens, total_tokens
    usage = {
        "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
        "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
        "total_tokens": getattr(response.usage, 'total_tokens', 0)
    }
    
    return usage


async def process_uploaded_files(
    files: List[UploadFile],
    user_id: str,
    message: Optional[str] = None
) -> tuple[List[dict], List[str], List[str]]:
    """
    Process uploaded files and return content parts and S3 URLs.
    
    Returns:
        Tuple of (content_parts, s3_urls, file_types)
    """
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")
    
    max_size = CONFIG.get("max_file_size_mb", 20) * 1024 * 1024
    content_parts = []
    s3_urls = []
    file_types = []
    
    for idx, file in enumerate(files):
        file_content = await file.read()
        filename = file.filename or f"file_{idx}"
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        mime_type = file.content_type or "application/octet-stream"
        
        # Validate file size
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File {filename} too large. Maximum size is {CONFIG.get('max_file_size_mb', 20)}MB"
            )
        
        # Upload to S3
        try:
            folder = "chat_images" if mime_type.startswith("image/") else "chat_files"
            result = storage.upload_bytes(
                data=file_content,
                filename=filename,
                user_id=user_id,
                folder=folder,
                content_type=mime_type
            )
            s3_urls.append(result.get("url"))
            file_types.append(file_ext)
            logger.info(f"File uploaded to S3: {result.get('url')}")
        except StorageError as e:
            logger.error(f"S3 upload failed for {filename}: {e}")
            raise HTTPException(status_code=503, detail=f"Upload failed for {filename}: {e}")
        
        # Build content part based on file type
        if mime_type.startswith("image/"):
            # Image: base64 encode for OpenAI
            image_base64 = base64.b64encode(file_content).decode("utf-8")
            content_parts.append({
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{image_base64}"
            })
        elif file_ext == 'pdf':
            # PDF: Send as file content block
            file_base64 = base64.b64encode(file_content).decode("utf-8")
            content_parts.append({
                "type": "input_file",
                "filename": filename,
                "file_data": f"data:application/pdf;base64,{file_base64}"
            })
        elif file_ext in ['docx', 'doc']:
            # DOCX: Extract text
            try:
                from docx import Document
                doc = Document(io.BytesIO(file_content))
                extracted_text = "\n".join([para.text for para in doc.paragraphs])
                content_parts.append({
                    "type": "input_text",
                    "text": f"[Document Content from {filename}]:\n{extracted_text}"
                })
            except Exception as e:
                logger.error(f"DOCX extraction failed: {e}")
                raise HTTPException(status_code=422, detail=f"Failed to extract DOCX content: {e}")
        elif file_ext in ['txt', 'md', 'json', 'yaml', 'yml', 'py', 'js', 'html', 'css', 'xml', 'csv']:
            # Text files
            try:
                text_content = file_content.decode('utf-8')
                content_parts.append({
                    "type": "input_text",
                    "text": f"[File Content from {filename}]:\n{text_content}"
                })
            except UnicodeDecodeError:
                raise HTTPException(status_code=422, detail=f"File {filename} is not a valid text file")
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type: {file_ext}. Supported: pdf, docx, txt, md, json, yaml, py, js, html, css, xml, csv, images"
            )
    
    return content_parts, s3_urls, file_types


def build_input_with_files(
    messages: List[ChatMessage],
    file_content_parts: List[dict],
    user_message: Optional[str] = None
) -> List[dict]:
    """Build OpenAI Chat Completions API messages with file content parts included."""
    # Use the new build_chat_messages function
    return build_chat_messages(messages, file_content_parts=file_content_parts)


def create_chat_response(
    response,
    model: str,
    conversation_id: str,
    user_id: str,
    message_count: int = 1
) -> ChatCompletionResponse:
    """Create a ChatCompletionResponse from OpenAI Chat Completions API response."""
    usage = extract_usage(response)
    cost = calculate_cost(usage)
    
    # Chat Completions API: content is in choices[0].message.content
    content = ""
    if response.choices and len(response.choices) > 0:
        content = response.choices[0].message.content or ""
    
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(
                    role="assistant",
                    content=content
                ),
                finish_reason="stop"
            )
        ],
        usage=UsageInfo(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"]
        ),
        conversation_id=conversation_id,
        user_id=user_id,
        cost=cost
    )


async def stream_chat_response(
    client: OpenAI,
    chat_request: ChatRequest,
    messages: List[dict],
    conversation_id: str,
    user_id: str,
    session_id: Optional[str],
    is_new_conversation: bool
):
    """Generate SSE stream for chat response using Chat Completions API."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    timestamp = int(time.time())
    model = chat_request.model or CONFIG["model"]
    
    full_text = []
    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    try:
        # Initial role message
        initial_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=timestamp,
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None
                )
            ]
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"
        
        # Stream from OpenAI using Chat Completions API
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
            top_p=chat_request.top_p,
            stream=True,
            stream_options={"include_usage": True}
        )
        
        for chunk in stream:
            # Chat Completions streaming: content is in chunk.choices[0].delta.content
            if chunk.choices and len(chunk.choices) > 0:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    full_text.append(delta_content)
                    
                    chunk_response = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=timestamp,
                        model=model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta=DeltaMessage(content=delta_content),
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {chunk_response.model_dump_json()}\n\n"
            
            # Check for usage in final chunk (some APIs include it)
            if hasattr(chunk, 'usage') and chunk.usage:
                usage_data["prompt_tokens"] = getattr(chunk.usage, 'prompt_tokens', 0)
                usage_data["completion_tokens"] = getattr(chunk.usage, 'completion_tokens', 0)
                usage_data["total_tokens"] = getattr(chunk.usage, 'total_tokens', 0)
        
        # Final chunk with usage
        final_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=timestamp,
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=usage_data["prompt_tokens"],
                completion_tokens=usage_data["completion_tokens"],
                total_tokens=usage_data["total_tokens"]
            )
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
        # Update database after stream completes
        cost = calculate_cost(usage_data)
        if db:
            try:
                db.update_conversation_usage(
                    conversation_id=conversation_id,
                    tokens_input=usage_data["prompt_tokens"],
                    tokens_output=usage_data["completion_tokens"],
                    cost=cost
                )
                update_session_after_chat(session_id, usage_data["total_tokens"], cost, is_new_conversation)
            except Exception as e:
                logger.error(f"Failed to update usage after stream: {e}")
                
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=timestamp,
            model=model,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaMessage(content=f"[ERROR] {str(e)}"),
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(ChatbotError)
async def chatbot_error_handler(request, exc: ChatbotError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorDetail(
                message=exc.message,
                type="chatbot_error",
                param=None,
                code=type(exc).__name__
            )
        ).model_dump()
    )


@app.exception_handler(RateLimitExceededError)
async def rate_limit_handler(request, exc: RateLimitExceededError):
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error=ErrorDetail(
                message=exc.message,
                type="rate_limit_error",
                param=None,
                code="rate_limit_exceeded"
            )
        ).model_dump(),
        headers={"Retry-After": str(int(exc.retry_after or 60))}
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error=ErrorDetail(
                message=exc.message,
                type="validation_error",
                param=getattr(exc, 'field', None),
                code="validation_error"
            )
        ).model_dump()
    )


@app.exception_handler(Exception)
async def generic_error_handler(request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(
                message="Internal server error",
                type="internal_error",
                param=None,
                code="internal_error"
            )
        ).model_dump()
    )


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def get_health():
    """Comprehensive health check."""
    result = health_check(include_details=True)
    status_code = 200 if result["status"] == "healthy" else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/health/live", tags=["Health"])
async def get_liveness():
    """Kubernetes liveness probe."""
    return liveness_check()


@app.get("/health/ready", tags=["Health"])
async def get_readiness():
    """Kubernetes readiness probe."""
    result = readiness_check()
    status_code = 200 if result["status"] == "ready" else 503
    return JSONResponse(content=result, status_code=status_code)


# =============================================================================
# Unified OpenAI-Compatible /chat Endpoint
# =============================================================================

@app.post("/chat", response_model=ChatCompletionResponse, tags=["Chat"])
async def chat(
    request: Request,
    client: OpenAI = Depends(get_openai)
):
    """
    Unified OpenAI-compatible chat endpoint.
    
    Supports:
    - Text-only chat (application/json)
    - Streaming (SSE with stream: true)
    - Images (base64 in JSON or multipart/form-data)
    - Documents (multipart/form-data)
    
    Content-Type handling:
    - application/json: Text and base64 images
    - multipart/form-data: File uploads (images, PDFs, DOCX, etc.)
    """
    content_type = request.headers.get("content-type", "")
    
    # Handle multipart form data (file uploads)
    if "multipart/form-data" in content_type:
        return await handle_multipart_chat(request, client)
    
    # Handle JSON requests
    return await handle_json_chat(request, client)


async def handle_json_chat(request: Request, client: OpenAI):
    """Handle JSON chat requests."""
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    
    user_id = chat_request.user_id or f"user_{uuid.uuid4().hex[:8]}"
    conversation_id = chat_request.conversation_id
    
    # Rate limiting
    limiter = get_rate_limiter(user_id)
    try:
        limiter.acquire(block=False)
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": str(int(e.retry_after or 60))}
        )
    
    # Sanitize input
    text_content = extract_text_from_content(chat_request.messages)
    try:
        _, warnings = sanitize_input(
            text_content,
            max_length=CONFIG.get("max_input_length", 10000)
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    # Session management
    session_id = get_or_create_session(user_id)
    is_new_conversation = False
    
    # Create conversation if needed
    if not conversation_id:
        is_new_conversation = True
        # Use local UUID for conversation ID (Chat Completions API doesn't manage conversations)
        conversation_id = f"conv_{uuid.uuid4().hex[:24]}"
        
        if db:
            try:
                db.create_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    metadata={"app": "terminal_chatbot_api"},
                    session_id=session_id
                )
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")
    
    # Build messages array for Chat Completions API
    system_prompt = chat_request.system_prompt or PROMPTS["system_prompt"]
    model = chat_request.model or CONFIG["model"]
    messages = build_chat_messages(chat_request.messages, system_prompt=system_prompt)
    
    # Handle streaming
    if chat_request.stream:
        return StreamingResponse(
            stream_chat_response(
                client=client,
                chat_request=chat_request,
                messages=messages,
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id,
                is_new_conversation=is_new_conversation
            ),
            media_type="text/event-stream",
            headers={"X-Conversation-ID": conversation_id}
        )
    
    # Non-streaming request using Chat Completions API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=chat_request.temperature,
            max_tokens=chat_request.max_tokens,
            top_p=chat_request.top_p
        )
    except (APIConnectionError, OpenAIRateLimitError, APITimeoutError) as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {type(e).__name__}")
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")
    
    # Extract usage and calculate cost
    usage = extract_usage(response)
    cost = calculate_cost(usage)
    
    # Update database
    if db:
        try:
            db.update_conversation_usage(
                conversation_id=conversation_id,
                tokens_input=usage["prompt_tokens"],
                tokens_output=usage["completion_tokens"],
                cost=cost
            )
        except Exception as e:
            logger.error(f"Failed to update usage: {e}")
    
    # Session tracking
    update_session_after_chat(session_id, usage["total_tokens"], cost, is_new_conversation)
    
    # Get message count
    message_count = 1
    if db:
        try:
            conv = db.get_conversation(conversation_id)
            if conv:
                message_count = conv.get("message_count", 1)
        except Exception:
            pass
    
    return create_chat_response(
        response=response,
        model=model,
        conversation_id=conversation_id,
        user_id=user_id,
        message_count=message_count
    )


async def handle_multipart_chat(request: Request, client: OpenAI):
    """Handle multipart form data chat requests with file uploads."""
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid form data: {e}")
    
    # Parse messages from JSON string
    messages_json = form.get("messages", "[]")
    try:
        messages_data = json.loads(messages_json)
        messages = [ChatMessage(**m) for m in messages_data]
    except (json.JSONDecodeError, Exception) as e:
        raise HTTPException(status_code=422, detail=f"Invalid messages JSON: {e}")
    
    # Get other parameters
    model = form.get("model", CONFIG["model"])
    stream = form.get("stream", "false").lower() == "true"
    temperature = float(form.get("temperature", 1.0))
    max_tokens = int(form.get("max_tokens")) if form.get("max_tokens") else None
    user_id = form.get("user_id") or f"user_{uuid.uuid4().hex[:8]}"
    conversation_id = form.get("conversation_id")
    system_prompt_override = form.get("system_prompt")
    
    # Get files (use hasattr check since UploadFile type may differ between fastapi/starlette)
    files: List[UploadFile] = []
    for key in form.keys():
        if key in ["files", "images"]:
            value = form.getlist(key)
            for f in value:
                if hasattr(f, 'filename') and hasattr(f, 'read'):
                    files.append(f)
    
    # Rate limiting
    limiter = get_rate_limiter(user_id)
    try:
        limiter.acquire(block=False)
    except RateLimitExceededError as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": str(int(e.retry_after or 60))}
        )
    
    # Sanitize input
    text_content = extract_text_from_content(messages)
    try:
        _, warnings = sanitize_input(
            text_content,
            max_length=CONFIG.get("max_input_length", 10000)
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    # Process files if any
    file_content_parts = []
    s3_urls = []
    file_types = []
    
    if files:
        file_content_parts, s3_urls, file_types = await process_uploaded_files(files, user_id)
    
    # Session management
    session_id = get_or_create_session(user_id)
    is_new_conversation = False
    
    # Create conversation if needed
    if not conversation_id:
        is_new_conversation = True
        # Use local UUID for conversation ID (Chat Completions API doesn't manage conversations)
        conversation_id = f"conv_{uuid.uuid4().hex[:24]}"
        
        if db:
            try:
                db_metadata = {"has_files": "true"} if file_types else {}
                db.create_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    metadata=db_metadata,
                    session_id=session_id
                )
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")
    
    # Build messages array for Chat Completions API (with file content parts)
    system_prompt = system_prompt_override or PROMPTS["system_prompt"]
    messages = build_chat_messages(messages, system_prompt=system_prompt, file_content_parts=file_content_parts)
    
    # Create ChatRequest for streaming
    chat_request = ChatRequest(
        model=model,
        messages=messages,
        stream=stream,
        temperature=temperature,
        max_tokens=max_tokens,
        user_id=user_id,
        conversation_id=conversation_id,
        system_prompt=system_prompt_override
    )
    
    # Handle streaming
    if stream:
        return StreamingResponse(
            stream_chat_response(
                client=client,
                chat_request=chat_request,
                messages=messages,
                conversation_id=conversation_id,
                user_id=user_id,
                session_id=session_id,
                is_new_conversation=is_new_conversation
            ),
            media_type="text/event-stream",
            headers={"X-Conversation-ID": conversation_id}
        )
    
    # Non-streaming request using Chat Completions API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0
        )
    except (APIConnectionError, OpenAIRateLimitError, APITimeoutError) as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {type(e).__name__}")
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")
    
    # Extract usage and calculate cost
    usage = extract_usage(response)
    cost = calculate_cost(usage)
    
    # Update database with file URLs
    if db:
        try:
            db.update_conversation_usage(
                conversation_id=conversation_id,
                tokens_input=usage["prompt_tokens"],
                tokens_output=usage["completion_tokens"],
                cost=cost,
                image_url=s3_urls[0] if s3_urls else None
            )
        except Exception as e:
            logger.error(f"Failed to update usage: {e}")
    
    # Session tracking
    update_session_after_chat(session_id, usage["total_tokens"], cost, is_new_conversation)
    
    # Get message count
    message_count = 1
    if db:
        try:
            conv = db.get_conversation(conversation_id)
            if conv:
                message_count = conv.get("message_count", 1)
        except Exception:
            pass
    
    return create_chat_response(
        response=response,
        model=model,
        conversation_id=conversation_id,
        user_id=user_id,
        message_count=message_count
    )


# =============================================================================
# Conversation Endpoints
# =============================================================================

@app.post("/conversations", response_model=ConversationResponse, tags=["Conversations"])
async def create_conversation(
    request: ConversationCreate
):
    """Create a new conversation."""
    # Chat Completions API doesn't manage conversations - we create local conversation ID
    conversation_id = f"conv_{uuid.uuid4().hex[:24]}"

    if db:
        try:
            db.create_conversation(
                conversation_id=conversation_id,
                user_id=request.user_id,
                metadata={
                    "app": "terminal_chatbot_api",
                    "created_at": datetime.now().isoformat(),
                    **(request.metadata or {})
                }
            )
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    return ConversationResponse(
        id=conversation_id,
        user_id=request.user_id,
        created_at=datetime.now().isoformat(),
        message_count=0
    )


@app.get("/conversations", response_model=List[ConversationResponse], tags=["Conversations"])
async def list_conversations(
    user_id: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List conversations for a user."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        conversations = db.list_conversations(user_id, limit=limit, offset=offset)
        return [
            ConversationResponse(
                id=c["id"],
                user_id=c["user_id"],
                created_at=c["created_at"].isoformat() if hasattr(c["created_at"], 'isoformat') else str(c["created_at"]),
                message_count=c.get("message_count", 0)
            )
            for c in conversations
        ]
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse, tags=["Conversations"])
async def get_conversation(conversation_id: str):
    """Get a conversation by ID."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        validate_conversation_id(conversation_id)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    conv = db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conv["id"],
        user_id=conv["user_id"],
        created_at=conv["created_at"].isoformat() if hasattr(conv["created_at"], 'isoformat') else str(conv["created_at"]),
        message_count=conv.get("message_count", 0)
    )


@app.delete("/conversations/{conversation_id}", tags=["Conversations"])
async def delete_conversation(conversation_id: str, soft: bool = True):
    """Delete a conversation."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        validate_conversation_id(conversation_id)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        db.delete_conversation(conversation_id, soft=soft)
        return {"success": True, "message": "Conversation deleted"}
    except DatabaseError as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail="Delete failed")


@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse], tags=["Conversations"])
async def get_messages(
    conversation_id: str,
    limit: int = Query(100, ge=1, le=1000),
    order: str = Query("asc", pattern="^(asc|desc)$")
):
    """Get messages for a conversation from database."""
    try:
        validate_conversation_id(conversation_id)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Chat Completions API doesn't store conversation history - get from database
    if not db:
        raise HTTPException(status_code=503, detail="Database not available - conversation history unavailable")
    
    try:
        messages = db.get_conversation_messages(
            conversation_id=conversation_id,
            limit=limit,
            order=order
        )
        
        return [
            MessageResponse(
                id=str(msg.get("id", uuid.uuid4())),
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                created_at=msg.get("created_at", datetime.now().isoformat()),
                tokens_input=msg.get("tokens_input", 0),
                tokens_output=msg.get("tokens_output", 0),
                cost=float(msg.get("cost", 0))
            )
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Failed to fetch messages from database: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch messages")


# =============================================================================
# User Endpoints
# =============================================================================

@app.get("/users/{user_id}/stats", response_model=UserStatsResponse, tags=["Users"])
async def get_user_stats(user_id: str, days: int = Query(30, ge=1, le=365)):
    """Get usage statistics for a user."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        stats = db.get_user_stats(user_id, days=days)
        return UserStatsResponse(
            total_conversations=stats.get("total_conversations", 0),
            total_messages=stats.get("total_messages", 0),
            total_tokens_input=stats.get("total_tokens_input", 0),
            total_tokens_output=stats.get("total_tokens_output", 0),
            total_cost=float(stats.get("total_cost", 0))
        )
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")


# =============================================================================
# Session Endpoints
# =============================================================================

@app.get("/sessions/{session_id}", response_model=SessionResponse, tags=["Sessions"])
async def get_session(session_id: str):
    """Get a session by ID."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        id=session["id"],
        user_id=session["user_id"],
        created_at=session["created_at"].isoformat() if hasattr(session["created_at"], 'isoformat') else str(session["created_at"]),
        last_activity=session["last_activity"].isoformat() if hasattr(session["last_activity"], 'isoformat') else str(session["last_activity"]),
        total_tokens=session.get("total_tokens", 0),
        total_cost=float(session.get("total_cost", 0)),
        is_active=session.get("is_active", True)
    )


@app.get("/users/{user_id}/sessions", response_model=List[SessionResponse], tags=["Sessions"])
async def get_user_sessions(user_id: str):
    """Get all sessions for a user (today's session)."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d')}"
    session = db.get_session(session_id)

    if not session:
        return []

    return [SessionResponse(
        id=session["id"],
        user_id=session["user_id"],
        created_at=session["created_at"].isoformat() if hasattr(session["created_at"], 'isoformat') else str(session["created_at"]),
        last_activity=session["last_activity"].isoformat() if hasattr(session["last_activity"], 'isoformat') else str(session["last_activity"]),
        total_tokens=session.get("total_tokens", 0),
        total_cost=float(session.get("total_cost", 0)),
        is_active=session.get("is_active", True)
    )]


@app.post("/sessions/{session_id}/end", tags=["Sessions"])
async def end_session(session_id: str):
    """End a session."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        db.end_session(session_id)
        return {"success": True, "message": "Session ended"}
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(status_code=500, detail="Failed to end session")


# =============================================================================
# File Endpoints
# =============================================================================

@app.post("/files/upload", response_model=FileUploadResponse, tags=["Files"])
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    folder: str = Form("uploads")
):
    """Upload a file to storage."""
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")

    # Validate file size
    max_size = CONFIG.get("max_file_size_mb", 20) * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {CONFIG.get('max_file_size_mb', 20)}MB"
        )

    try:
        result = storage.upload_bytes(
            data=content,
            filename=file.filename,
            user_id=user_id,
            folder=folder,
            content_type=file.content_type
        )
        return FileUploadResponse(
            success=True,
            key=result.get("key"),
            url=result.get("url"),
            size=len(content)
        )
    except StorageError as e:
        logger.error(f"S3 upload failed: {e}")
        raise HTTPException(status_code=503, detail=f"Upload failed: {e}")


@app.get("/files", tags=["Files"])
async def list_files(
    user_id: str = Query(...),
    folder: str = Query("uploads"),
    limit: int = Query(100, ge=1, le=1000)
):
    """List files for a user."""
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")

    try:
        files = storage.list_files(user_id, folder, limit)
        return {"files": files}
    except StorageError as e:
        logger.error(f"S3 list failed: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to list files: {e}")


@app.delete("/files/{file_key:path}", tags=["Files"])
async def delete_file(file_key: str):
    """Delete a file from storage."""
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")

    try:
        storage.delete_file(file_key)
        return {"success": True, "message": "File deleted"}
    except StorageError as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file")


# =============================================================================
# Config Endpoint
# =============================================================================

@app.get("/config", tags=["Config"])
async def get_config():
    """Get current configuration (non-sensitive)."""
    safe_config = {
        "model": CONFIG.get("model"),
        "max_input_length": CONFIG.get("max_input_length"),
        "max_file_size_mb": CONFIG.get("max_file_size_mb"),
        "rate_limit_per_minute": CONFIG.get("rate_limit_per_minute"),
        "stream_responses": CONFIG.get("stream_responses"),
    }
    return {"config": safe_config}


# =============================================================================
# Root
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root - basic info."""
    return {
        "name": "Terminal Chatbot API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "conversations": "/conversations",
            "files": "/files",
            "health": "/health",
            "docs": "/docs"
        },
        "features": {
            "openai_compatible": True,
            "streaming": True,
            "multimodal": True,
            "file_uploads": True
        }
    }


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        log_level="info"
    )
