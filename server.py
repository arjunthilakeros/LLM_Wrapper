"""
FastAPI Server for Terminal Chatbot - OpenAI Responses API Only
REST API with PostgreSQL database and S3 storage integration.

Uses OpenAI Responses API exclusively for all chat operations.
"""

import os
import uuid
import base64
import json
import time
import io
from datetime import datetime
from typing import Optional, List, Union, Literal, Any, Dict
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, Request
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
from health import health_check

# Import Responses API modules
from responses_client import ResponsesAPIClient, create_responses_client, extract_output_text, extract_usage, generate_title

# Import Context Manager for Summary + Window strategy
from context_manager import ContextManager, ContextConfig

# All routes are defined in this file - no separate routes module needed

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
responses_client: Optional[ResponsesAPIClient] = None  # ResponsesAPIClient instance
context_manager: Optional[ContextManager] = None  # Context manager for summary+window
context_config: Optional[ContextConfig] = None  # Context configuration
rate_limiters: dict = {}  # Per-user rate limiters


# =============================================================================
# Pydantic Models (Responses API only)
# =============================================================================

class MessageContent(BaseModel):
    """Content part for multimodal messages."""
    type: Literal["text", "image_url", "file"] = "text"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    file: Optional[Dict[str, str]] = None


class ChatRequest(BaseModel):
    """Request model for Responses API chat endpoint."""
    input: Union[str, List[MessageContent]] = Field(
        ..., 
        description="User input - text string or multimodal content"
    )
    model: str = Field(
        default="gpt-4o", 
        description="Model to use for the response"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Existing conversation ID (optional)"
    )
    instructions: Optional[str] = Field(
        default=None,
        description="System instructions for the assistant"
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming response"
    )
    temperature: Optional[float] = Field(
        default=1.0,
        ge=0, le=2,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate"
    )
    top_p: Optional[float] = Field(
        default=1.0,
        ge=0, le=1,
        description="Nucleus sampling parameter"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for rate limiting"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tools available to the assistant"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Tool choice configuration"
    )
    previous_response_id: Optional[str] = Field(
        default=None,
        description="Previous response ID for chaining"
    )

    @field_validator('input')
    @classmethod
    def validate_input(cls, v):
        if isinstance(v, str):
            if len(v) > 100000:  # Reasonable limit
                raise ValueError("Input too long (max 100,000 characters)")
        elif isinstance(v, list):
            if not v:
                raise ValueError("Input content list cannot be empty")
        return v


class UsageInfo(BaseModel):
    """Token usage information."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ResponseOutput(BaseModel):
    """Output item from Responses API."""
    type: str  # "message", "file_search", "function_call", etc.
    id: Optional[str] = None
    role: Optional[str] = None  # "assistant" for messages
    content: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for Responses API chat endpoint."""
    id: str = Field(..., description="Response ID from OpenAI")
    object: str = Field(default="response", description="Object type")
    created_at: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    output: List[ResponseOutput] = Field(default=[], description="Output items")
    output_text: Optional[str] = Field(
        default=None,
        description="Helper: concatenated text output"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID"
    )
    title: Optional[str] = Field(
        default=None,
        description="Conversation title"
    )
    message_count: int = Field(
        default=0,
        description="Total messages in conversation"
    )
    previous_response_id: Optional[str] = Field(
        default=None,
        description="Previous response ID in chain"
    )
    usage: Optional[UsageInfo] = Field(default=None, description="Token usage")
    cost: Optional[float] = Field(default=None, description="Estimated cost in USD")


class ConversationCreate(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)
    metadata: Optional[dict] = None
    title: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = Field(None, max_length=500)
    metadata: Optional[dict] = None


class ConversationResponse(BaseModel):
    id: str
    user_id: str
    created_at: str
    message_count: int = 0
    title: Optional[str] = None
    summary: Optional[str] = None
    total_tokens: int = 0
    total_cost: float = 0
    metadata: Optional[dict] = None


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    tokens_input: int = 0
    tokens_output: int = 0
    cost: float = 0


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    checks: dict
    uptime: dict


class ErrorDetail(BaseModel):
    """Error detail model."""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: ErrorDetail


# =============================================================================
# Lifespan (Startup/Shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    global db, storage, responses_client, context_manager, context_config

    logger.info("Starting API server...")

    # Initialize Responses API client
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set!")
        raise ConfigurationError("OPENAI_API_KEY environment variable is required")

    try:
        responses_client = create_responses_client(
            api_key=OPENAI_API_KEY,
            timeout=CONFIG.get("api_timeout", 30),
            max_retries=CONFIG.get("api_max_retries", 3)
        )
        logger.info("Responses API client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Responses API client: {e}")
        raise ConfigurationError(f"Failed to initialize Responses API client: {e}")

    # Initialize Context Manager for Summary + Window strategy
    context_config = ContextConfig.from_dict(CONFIG)
    logger.info(f"Context management mode: {context_config.mode}")

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

    # Initialize Context Manager (after database so it can use it)
    context_manager = ContextManager(
        config=context_config,
        responses_client=responses_client,
        database=db,
        data_dir=BASE_DIR / CONFIG.get("data_dir", "./data"),
    )
    logger.info("Context manager initialized")

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
    description="Production-ready chatbot API with OpenAI Responses API, PostgreSQL, and S3 integration.",
    version="3.0.0",
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


def get_responses_client():
    """Dependency to get Responses API client."""
    if not responses_client:
        raise HTTPException(status_code=503, detail="Responses API client not initialized")
    return responses_client


# =============================================================================
# Helper Functions
# =============================================================================

def build_input_content(input_data: Union[str, List[MessageContent]]) -> Union[str, List[Dict]]:
    """Build input content for OpenAI Responses API."""
    if isinstance(input_data, str):
        return input_data

    # Convert MessageContent list to dict format for Responses API
    # Multimodal content needs to be wrapped in a message with role
    content_list = []
    for item in input_data:
        if item.type == "text" and item.text:
            content_list.append({"type": "input_text", "text": item.text})
        elif item.type == "image_url" and item.image_url:
            content_list.append({
                "type": "input_image",
                "image_url": item.image_url.get("url", "")
            })
        elif item.type == "file" and item.file:
            content_list.append({
                "type": "input_file",
                "file_data": item.file.get("data", ""),
                "filename": item.file.get("name", "document")
            })

    # Wrap in message format for Responses API (required for multimodal)
    return [{"role": "user", "content": content_list}]


def calculate_cost(usage: Dict[str, int], model: str = "gpt-4o") -> float:
    """Calculate cost from usage based on model pricing."""
    pricing = CONFIG.get("pricing", {"input_per_1k": 0.0025, "output_per_1k": 0.01})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    
    cost = (
        (input_tokens / 1000) * pricing["input_per_1k"] +
        (output_tokens / 1000) * pricing["output_per_1k"]
    )
    return round(cost, 6)


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


# =============================================================================
# Chat Endpoint (Responses API)
# =============================================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(
    request: ChatRequest,
    client: ResponsesAPIClient = Depends(get_responses_client)
):
    """
    Main chat endpoint using OpenAI Responses API.

    Supports:
    - Text-only chat
    - Streaming (SSE with stream: true)
    - Multimodal input (images, files)
    - Conversation management
    - Summary + Window context optimization (when mode: summary_window)

    Context Management:
    - mode="full": Uses conversation_id, OpenAI handles full history
    - mode="summary_window": Manually builds [Summary] + [Window] context

    If no conversation_id is provided, creates a new conversation.
    """
    user_id = request.user_id or f"user_{uuid.uuid4().hex[:8]}"

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

    # Sanitize input if it's a string
    if isinstance(request.input, str):
        try:
            sanitized_input, _ = sanitize_input(
                request.input,
                max_length=CONFIG.get("max_input_length", 10000)
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
    else:
        sanitized_input = request.input

    # Build input content
    input_content = build_input_content(sanitized_input)

    # Get conversation data for context strategy decision
    message_count = 0
    conv_data = None
    is_new_conversation = request.conversation_id is None

    if db and request.conversation_id:
        try:
            conv_data = db.get_conversation(request.conversation_id)
            if conv_data:
                message_count = conv_data.get("message_count", 0) or 0
        except Exception:
            pass

    # Determine context strategy
    use_summary_window = (
        context_config and
        context_config.mode == "summary_window" and
        request.conversation_id and
        message_count >= context_config.summarize_after_messages
    )

    context_metadata = {"mode": "full"}

    # Prepare request parameters
    params = {
        "model": request.model or CONFIG["model"],
        "store": True,  # Always store for history retrieval
    }

    if use_summary_window and context_manager:
        # Summary + Window mode: manually build context
        try:
            context_result = context_manager.build_context_for_api(
                conversation_id=request.conversation_id,
                new_message=sanitized_input if isinstance(sanitized_input, str) else str(sanitized_input),
                system_prompt=request.instructions or PROMPTS.get("system_prompt"),
                message_count=message_count
            )

            params["input"] = context_result["input"]
            # Don't use conversation_id in summary_window mode
            context_metadata = context_result["metadata"]

            logger.info(
                f"Chat using summary_window: {context_metadata.get('total_tokens', 0)} tokens "
                f"(summary={context_metadata.get('summary_tokens', 0)}, "
                f"window={context_metadata.get('window_tokens', 0)})"
            )
        except Exception as e:
            logger.warning(f"Context manager failed, falling back to full mode: {e}")
            params["input"] = input_content
            if request.conversation_id:
                params["conversation_id"] = request.conversation_id
    else:
        # Full mode: let OpenAI handle context
        params["input"] = input_content
        if request.conversation_id:
            params["conversation_id"] = request.conversation_id

    if request.instructions:
        params["instructions"] = request.instructions
    elif PROMPTS.get("system_prompt"):
        params["instructions"] = PROMPTS["system_prompt"]
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.max_tokens is not None:
        params["max_output_tokens"] = request.max_tokens
    if request.top_p is not None:
        params["top_p"] = request.top_p
    if request.tools:
        params["tools"] = request.tools
    if request.tool_choice:
        params["tool_choice"] = request.tool_choice
    if request.previous_response_id:
        params["previous_response_id"] = request.previous_response_id

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            stream_chat_response(client, params, user_id, context_metadata),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-ID": request.conversation_id or "",
                "X-Context-Mode": context_metadata.get("mode", "full")
            }
        )

    # Non-streaming request
    try:
        response = client.create_response(**params)
    except ChatbotAPIError as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=502, detail=f"API error: {e.message}")
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
    
    # Extract usage
    usage_data = extract_usage(response) or {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    # Calculate cost
    cost = calculate_cost(usage_data, request.model or CONFIG["model"])
    
    # Format output
    output_items = []
    for item in response.output:
        item_dict = {
            "type": getattr(item, "type", "unknown"),
            "id": getattr(item, "id", None),
            "role": getattr(item, "role", None),
            "content": [],
            "status": getattr(item, "status", None)
        }
        
        # Extract content
        content = getattr(item, "content", [])
        if content:
            for content_part in content:
                if hasattr(content_part, "model_dump"):
                    item_dict["content"].append(content_part.model_dump())
                elif isinstance(content_part, dict):
                    item_dict["content"].append(content_part)
                else:
                    item_dict["content"].append({"type": "text", "text": str(content_part)})
        
        output_items.append(ResponseOutput(**item_dict))
    
    # Extract text for convenience
    output_text = extract_output_text(response)
    
    # Get conversation ID from response if available
    conversation_id = getattr(response, "conversation_id", None) or request.conversation_id

    # Track title for response
    title = None
    updated_message_count = message_count

    # Update database with usage and handle title/summary
    if db and conversation_id:
        try:
            # Check if this is a new conversation that needs to be created locally
            if not conv_data:
                # Create local record for new OpenAI conversation
                db.create_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    metadata={"openai_conversation": True}
                )
                is_new_conversation = True

            # Update usage and increment message count
            update_result = db.update_conversation_usage(
                conversation_id=conversation_id,
                tokens_input=usage_data["input_tokens"],
                tokens_output=usage_data["output_tokens"],
                cost=cost,
                increment_message_count=True
            )

            if update_result:
                updated_message_count = update_result.get("message_count", message_count + 1)

            # Generate title on first message
            if is_new_conversation or (conv_data and not conv_data.get("title")):
                try:
                    first_message = sanitized_input if isinstance(sanitized_input, str) else str(sanitized_input)
                    title = generate_title(first_message, client)
                    db.update_title(conversation_id, title)
                    logger.info(f"Generated title for {conversation_id}: {title}")
                except Exception as e:
                    logger.warning(f"Failed to generate title: {e}")
            else:
                title = conv_data.get("title") if conv_data else None

            # Check if summary needs update (every N messages)
            summary_interval = context_config.summary_update_interval if context_config else 10
            if db.should_update_summary(conversation_id, interval=summary_interval):
                # Summary update will happen in context_manager when needed
                logger.debug(f"Summary update needed for {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to update conversation: {e}")

    return ChatResponse(
        id=response.id,
        object="response",
        created_at=getattr(response, "created_at", int(time.time())),
        model=response.model,
        output=output_items,
        output_text=output_text if output_text else None,
        conversation_id=conversation_id,
        title=title,
        message_count=updated_message_count,
        previous_response_id=getattr(response, "previous_response_id", None),
        usage=UsageInfo(**usage_data),
        cost=cost
    )


async def stream_chat_response(
    client: ResponsesAPIClient,
    params: dict,
    user_id: str,
    context_metadata: dict = None
):
    """Generate SSE stream for chat response using Responses API."""
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    timestamp = int(time.time())
    context_metadata = context_metadata or {"mode": "full"}
    
    try:
        # Create streaming response
        stream = client.create_response_stream_raw(**params)
        
        full_text = []
        usage_data = {"input_tokens": 0, "output_tokens": 0}
        conversation_id = None
        
        for event in stream:
            event_type = getattr(event, "type", "")
            
            if event_type == "response.output_text.delta":
                # Text delta
                delta = getattr(event, "delta", "")
                if delta:
                    full_text.append(delta)
                    chunk = {
                        "id": response_id,
                        "object": "response.chunk",
                        "created_at": timestamp,
                        "delta": {"type": "text", "text": delta}
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    
            elif event_type == "response.output_item.added":
                # New output item
                item = getattr(event, "item", {})
                chunk = {
                    "id": response_id,
                    "object": "response.chunk",
                    "created_at": timestamp,
                    "delta": {"type": "item_added", "item": item}
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
            elif event_type == "response.completed":
                # Final response with usage
                response = getattr(event, "response", {})
                usage = getattr(response, "usage", None)
                if usage:
                    usage_data = {
                        "input_tokens": getattr(usage, "input_tokens", 0),
                        "output_tokens": getattr(usage, "output_tokens", 0)
                    }
                
                # Get conversation ID
                conversation_id = getattr(response, "conversation_id", None)
                
                # Send final chunk
                final_chunk = {
                    "id": response_id,
                    "object": "response.completed",
                    "created_at": timestamp,
                    "conversation_id": conversation_id,
                    "usage": usage_data,
                    "context_mode": context_metadata.get("mode", "full"),
                    "context_tokens": context_metadata.get("total_tokens", 0),
                    "done": True
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                break
                
            elif event_type == "error":
                error_msg = getattr(event, "message", "Unknown streaming error")
                logger.error(f"Streaming error: {error_msg}")
                error_chunk = {
                    "id": response_id,
                    "object": "response.error",
                    "created_at": timestamp,
                    "error": error_msg
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                break
        
    except Exception as e:
        logger.error(f"Stream generation error: {e}")
        error_chunk = {
            "id": response_id,
            "object": "response.error",
            "created_at": timestamp,
            "error": str(e)
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# =============================================================================
# Conversation Endpoints
# =============================================================================

@app.post("/conversations", response_model=ConversationResponse, tags=["Conversations"])
async def create_conversation(
    request: ConversationCreate,
    client: ResponsesAPIClient = Depends(get_responses_client)
):
    """Create a new conversation using OpenAI Conversations API."""
    user_id = request.user_id

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

    try:
        # Create conversation in OpenAI
        conversation = client.create_conversation(
            metadata={
                "user_id": user_id,
                "title": request.title or "",
                **(request.metadata or {})
            }
        )

        conversation_id = conversation.id
        created_at = getattr(conversation, "created_at", int(time.time()))

        # Store in local database if available
        if db:
            try:
                db.create_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    title=request.title,
                    metadata={
                        "openai_conversation": True,
                        **(request.metadata or {})
                    }
                )
            except Exception as e:
                logger.error(f"Failed to store conversation in DB: {e}")

        return ConversationResponse(
            id=conversation_id,
            user_id=user_id,
            created_at=datetime.fromtimestamp(created_at).isoformat(),
            message_count=0,
            title=request.title,
            summary=None,
            total_tokens=0,
            total_cost=0,
            metadata=request.metadata
        )

    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to create conversation: {str(e)}")


@app.get("/conversations", response_model=List[ConversationResponse], tags=["Conversations"])
async def list_conversations(
    user_id: str = Query(..., min_length=1),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List conversations for a user from database."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        conversations = db.list_conversations(user_id, limit=limit, offset=offset)
        return [
            ConversationResponse(
                id=c["id"],
                user_id=c["user_id"],
                created_at=c["created_at"].isoformat() if hasattr(c["created_at"], 'isoformat') else str(c["created_at"]),
                message_count=c.get("message_count", 0) or 0,
                title=c.get("title"),
                summary=c.get("summary"),
                total_tokens=(c.get("total_tokens_input", 0) or 0) + (c.get("total_tokens_output", 0) or 0),
                total_cost=float(c.get("total_cost", 0) or 0),
                metadata=c.get("metadata")
            )
            for c in conversations
        ]
    except DatabaseError as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/conversations/{conversation_id}", response_model=ConversationResponse, tags=["Conversations"])
async def get_conversation(conversation_id: str):
    """Get a conversation by ID from database."""
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
        message_count=conv.get("message_count", 0) or 0,
        title=conv.get("title"),
        summary=conv.get("summary"),
        total_tokens=(conv.get("total_tokens_input", 0) or 0) + (conv.get("total_tokens_output", 0) or 0),
        total_cost=float(conv.get("total_cost", 0) or 0),
        metadata=conv.get("metadata")
    )


@app.patch("/conversations/{conversation_id}", response_model=ConversationResponse, tags=["Conversations"])
async def update_conversation(
    conversation_id: str,
    request: ConversationUpdate
):
    """Update a conversation (title, metadata)."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        validate_conversation_id(conversation_id)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    conv = db.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Update title if provided
    if request.title is not None:
        db.update_title(conversation_id, request.title)

    # Update metadata if provided
    if request.metadata is not None:
        db.update_conversation(conversation_id, metadata=request.metadata)

    # Fetch updated conversation
    conv = db.get_conversation(conversation_id)

    return ConversationResponse(
        id=conv["id"],
        user_id=conv["user_id"],
        created_at=conv["created_at"].isoformat() if hasattr(conv["created_at"], 'isoformat') else str(conv["created_at"]),
        message_count=conv.get("message_count", 0) or 0,
        title=conv.get("title"),
        summary=conv.get("summary"),
        total_tokens=(conv.get("total_tokens_input", 0) or 0) + (conv.get("total_tokens_output", 0) or 0),
        total_cost=float(conv.get("total_cost", 0) or 0),
        metadata=conv.get("metadata")
    )


@app.delete("/conversations/{conversation_id}", tags=["Conversations"])
async def delete_conversation(
    conversation_id: str, 
    soft: bool = Query(True),
    client: ResponsesAPIClient = Depends(get_responses_client)
):
    """Delete a conversation from both OpenAI and local database."""
    if not db:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        validate_conversation_id(conversation_id)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        # Try to delete from OpenAI
        try:
            client.delete_conversation(conversation_id)
        except Exception as e:
            logger.warning(f"Failed to delete conversation from OpenAI: {e}")
        
        # Delete from local database
        db.delete_conversation(conversation_id, soft=soft)
        return {"success": True, "message": "Conversation deleted"}
    except DatabaseError as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail="Delete failed")


@app.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse], tags=["Conversations"])
async def get_messages(
    conversation_id: str,
    limit: int = Query(100, ge=1, le=1000),
    order: str = Query("asc", pattern="^(asc|desc)$"),
    client: ResponsesAPIClient = Depends(get_responses_client)
):
    """Get messages for a conversation from OpenAI Conversations API."""
    try:
        validate_conversation_id(conversation_id)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        # Fetch items from OpenAI Conversations API
        items_response = client.list_conversation_items(
            conversation_id=conversation_id,
            limit=limit,
            order=order
        )
        
        # Extract items from response
        items = []
        if hasattr(items_response, "data"):
            items = items_response.data
        elif isinstance(items_response, list):
            items = items_response
        
        # Convert to MessageResponse format
        messages = []
        for item in items:
            item_type = getattr(item, "type", "")
            
            if item_type == "message":
                role = getattr(item, "role", "unknown")
                content_parts = getattr(item, "content", [])
                
                # Extract text from content
                text_parts = []
                for part in content_parts:
                    if hasattr(part, "type"):
                        if part.type in ["input_text", "output_text", "text"]:
                            text_parts.append(getattr(part, "text", ""))
                        elif part.type == "input_image":
                            text_parts.append("[Image]")
                        elif part.type == "input_file":
                            text_parts.append(f"[File: {getattr(part, 'filename', 'document')}]")
                
                messages.append(MessageResponse(
                    id=getattr(item, "id", str(uuid.uuid4())),
                    role=role,
                    content="\n".join(text_parts) if text_parts else "",
                    created_at=getattr(item, "created_at", datetime.now().isoformat()),
                    tokens_input=0,
                    tokens_output=0,
                    cost=0
                ))
        
        return messages
        
    except Exception as e:
        logger.error(f"Failed to fetch messages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch messages: {str(e)}")


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
        "version": "3.0.0",
        "status": "running",
        "api": "OpenAI Responses API",
        "endpoints": {
            "chat": "/chat (Responses API)",
            "conversations": "/conversations",
            "health": "/health",
            "docs": "/docs"
        },
        "features": {
            "openai_compatible": True,
            "streaming": True,
            "multimodal": True,
            "file_uploads": True,
            "responses_api": responses_client is not None
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
