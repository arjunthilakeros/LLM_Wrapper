"""
FastAPI Server for Terminal Chatbot
REST API with PostgreSQL database and S3 storage integration.
"""

import os
import uuid
import base64
import json
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

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
# Pydantic Models
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=50000)
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    stream: bool = False

class ChatResponse(BaseModel):
    text: str
    conversation_id: str
    user_id: str
    message_number: int
    usage: dict
    cost: float

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

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class ImageChatResponse(BaseModel):
    text: str
    conversation_id: str
    user_id: str
    message_number: int
    usage: dict
    cost: float
    image_urls: Optional[List[str]] = None  # S3 URLs of uploaded images
    image_count: int = 1


class FileChatResponse(BaseModel):
    text: str
    conversation_id: str
    user_id: str
    message_number: int
    usage: dict
    cost: float
    file_url: Optional[str] = None  # S3 URL of uploaded file
    file_type: str  # pdf, docx, txt, etc.


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

    # Initialize Storage (S3 required for production)
    if not S3_ENABLED:
        raise ConfigurationError("S3_ENABLED must be true for production. Set S3_ENABLED=true in environment.")

    if not STORAGE_AVAILABLE:
        raise ConfigurationError("S3 storage unavailable. Install boto3: pip install boto3")

    storage = S3Storage.initialize()
    logger.info("S3 storage initialized")

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
    description="Production-ready chatbot API with OpenAI, PostgreSQL, and S3 integration",
    version="1.0.0",
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



# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(ChatbotError)
async def chatbot_error_handler(request, exc: ChatbotError):
    return JSONResponse(
        status_code=400,
        content={"error": exc.message, "detail": str(exc.details), "code": type(exc).__name__}
    )

@app.exception_handler(RateLimitExceededError)
async def rate_limit_handler(request, exc: RateLimitExceededError):
    return JSONResponse(
        status_code=429,
        content={
            "error": exc.message,
            "retry_after": exc.retry_after,
            "code": "RateLimitExceeded"
        },
        headers={"Retry-After": str(int(exc.retry_after or 60))}
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": exc.message, "field": exc.field, "code": "ValidationError"}
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
# Chat Endpoints
# =============================================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, client: OpenAI = Depends(get_openai)):
    """Send a chat message and get a response."""
    user_id = request.user_id or f"user_{uuid.uuid4().hex[:8]}"
    conversation_id = request.conversation_id

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
    try:
        message, warnings = sanitize_input(
            request.message,
            max_length=CONFIG.get("max_input_length", 10000)
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Get or create session first (needed for linking)
    session_id = get_or_create_session(user_id)
    is_new_conversation = False

    # Create conversation if needed
    if not conversation_id:
        is_new_conversation = True
        conv = client.conversations.create(
            metadata={
                "app": "terminal_chatbot_api",
                "user_id": user_id,
                "created_at": datetime.now().isoformat()
            }
        )
        conversation_id = conv.id

        # Save to database with session_id
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

    # Send message
    try:
        response = client.responses.create(
            model=CONFIG["model"],
            input=message,
            conversation=conversation_id,
            instructions=PROMPTS["system_prompt"]
        )
    except (APIConnectionError, OpenAIRateLimitError, APITimeoutError) as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {type(e).__name__}")

    # Extract usage
    usage = {
        "input": getattr(response.usage, 'input_tokens', 0) if response.usage else 0,
        "output": getattr(response.usage, 'output_tokens', 0) if response.usage else 0,
        "total": getattr(response.usage, 'total_tokens', 0) if response.usage else 0
    }

    # Calculate cost
    pricing = CONFIG["pricing"]
    cost = round(
        (usage["input"] / 1000) * pricing["input_per_1k"] +
        (usage["output"] / 1000) * pricing["output_per_1k"],
        6
    )

    # Update usage stats in database (OpenAI stores the actual messages)
    if db:
        try:
            db.update_conversation_usage(
                conversation_id=conversation_id,
                tokens_input=usage["input"],
                tokens_output=usage["output"],
                cost=cost
            )
        except Exception as e:
            logger.error(f"Failed to update usage: {e}")

    # Session tracking (increment conversation_count if new)
    update_session_after_chat(session_id, usage["total"], cost, is_new_conversation)

    # Get message count
    message_count = 1
    if db:
        try:
            conv = db.get_conversation(conversation_id)
            if conv:
                message_count = conv.get("message_count", 1)
        except Exception:
            pass

    return ChatResponse(
        text=response.output_text,
        conversation_id=conversation_id,
        user_id=user_id,
        message_number=message_count,
        usage=usage,
        cost=cost
    )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest, client: OpenAI = Depends(get_openai)):
    """Send a chat message and stream the response."""
    user_id = request.user_id or f"user_{uuid.uuid4().hex[:8]}"
    conversation_id = request.conversation_id

    # Rate limiting
    limiter = get_rate_limiter(user_id)
    try:
        limiter.acquire(block=False)
    except RateLimitExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))

    # Sanitize input
    try:
        message, _ = sanitize_input(request.message, max_length=CONFIG.get("max_input_length", 10000))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Create conversation if needed
    if not conversation_id:
        conv = client.conversations.create(
            metadata={"app": "terminal_chatbot_api", "user_id": user_id}
        )
        conversation_id = conv.id
        if db:
            try:
                db.create_conversation(conversation_id, user_id)
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")

    # Track session for the generate function
    session_id = get_or_create_session(user_id)
    is_new_conversation = not request.conversation_id

    async def generate():
        try:
            stream = client.responses.create(
                model=CONFIG["model"],
                input=message,
                conversation=conversation_id,
                instructions=PROMPTS["system_prompt"],
                stream=True
            )

            full_text = []
            usage_data = {"input": 0, "output": 0, "total": 0}

            for event in stream:
                if hasattr(event, 'type'):
                    if event.type == 'response.output_text.delta':
                        delta = getattr(event, 'delta', '')
                        full_text.append(delta)
                        yield f"data: {delta}\n\n"
                    elif event.type == 'response.completed':
                        # Try to extract usage from completed event
                        if hasattr(event, 'usage'):
                            usage_data["input"] = getattr(event.usage, 'input_tokens', 0) or getattr(event.usage, 'prompt_tokens', 0)
                            usage_data["output"] = getattr(event.usage, 'output_tokens', 0) or getattr(event.usage, 'completion_tokens', 0)
                            usage_data["total"] = getattr(event.usage, 'total_tokens', 0)

                        yield f"data: [DONE]\n\n"

            # Calculate cost
            pricing = CONFIG.get("pricing", {"input_per_1k": 0.0025, "output_per_1k": 0.01})
            cost = round(
                (usage_data["input"] / 1000) * pricing["input_per_1k"] +
                (usage_data["output"] / 1000) * pricing["output_per_1k"],
                6
            )

            # Update usage stats (OpenAI stores actual messages)
            if db:
                try:
                    db.update_conversation_usage(
                        conversation_id=conversation_id,
                        tokens_input=usage_data["input"],
                        tokens_output=usage_data["output"],
                        cost=cost
                    )
                    # Also update session
                    update_session_after_chat(session_id, usage_data["total"], cost, is_new_conversation)
                except Exception as e:
                    logger.error(f"Failed to update usage: {e}")

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Conversation-ID": conversation_id}
    )


@app.post("/chat/image", response_model=ImageChatResponse, tags=["Chat"])
async def chat_with_image(
    message: str = Form(...),
    images: List[UploadFile] = File(..., description="One or more images"),
    conversation_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    client: OpenAI = Depends(get_openai)
):
    """Send a chat message with one or more images to GPT-4 Vision."""
    user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"

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
    try:
        message, warnings = sanitize_input(
            message,
            max_length=CONFIG.get("max_input_length", 10000)
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")

    max_size = CONFIG.get("max_file_size_mb", 20) * 1024 * 1024
    s3_urls = []
    image_content_list = []

    # Process all images
    for idx, image in enumerate(images):
        image_content = await image.read()

        # Validate file size
        if len(image_content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Image {idx+1} too large. Maximum size is {CONFIG.get('max_file_size_mb', 20)}MB"
            )

        mime_type = image.content_type or "image/jpeg"
        image_base64 = base64.b64encode(image_content).decode("utf-8")

        # Upload to S3
        try:
            result = storage.upload_bytes(
                data=image_content,
                filename=image.filename or f"{uuid.uuid4().hex}.jpg",
                user_id=user_id,
                folder="chat_images",
                content_type=mime_type
            )
            s3_urls.append(result.get("url"))
            logger.info(f"Image {idx+1} uploaded to S3: {result.get('url')}")
        except StorageError as e:
            logger.error(f"S3 upload failed for image {idx+1}: {e}")
            raise HTTPException(status_code=503, detail=f"Image {idx+1} upload failed: {e}")

        image_content_list.append({
            "base64": image_base64,
            "mime_type": mime_type
        })

    # Create conversation if needed (with session linking)
    session_id = get_or_create_session(user_id)
    is_new_conversation = False

    if not conversation_id:
        is_new_conversation = True
        conv = client.conversations.create(
            metadata={
                "app": "terminal_chatbot_api",
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "has_images": "true",
                "image_count": str(len(images))
            }
        )
        conversation_id = conv.id

        if db:
            try:
                db.create_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    metadata={"has_images": "true", "image_count": str(len(images))},
                    session_id=session_id
                )
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")

    # Build content array with text + all images
    content_parts = [{"type": "input_text", "text": message}]
    for img_data in image_content_list:
        content_parts.append({
            "type": "input_image",
            "image_url": f"data:{img_data['mime_type']};base64,{img_data['base64']}"
        })

    input_content = [
        {
            "type": "message",
            "role": "user",
            "content": content_parts
        }
    ]

    # Call OpenAI (Stateful)
    try:
        response = client.responses.create(
            model="gpt-4o",
            conversation=conversation_id,
            input=input_content,
            instructions=PROMPTS["system_prompt"],
            max_output_tokens=4096
        )
    except (APIConnectionError, OpenAIRateLimitError, APITimeoutError) as e:
        logger.error(f"OpenAI Vision API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {type(e).__name__}")

    # Extract usage and calculate cost
    usage = {
        "input": getattr(response.usage, 'input_tokens', 0) if response.usage else 0,
        "output": getattr(response.usage, 'output_tokens', 0) if response.usage else 0,
        "total": getattr(response.usage, 'total_tokens', 0) if response.usage else 0
    }

    # Fallback if usage is missing or different structure
    if usage["total"] == 0 and hasattr(response, 'usage_metadata'):
        usage = {
            "input": getattr(response.usage_metadata, 'prompt_tokens', 0),
            "output": getattr(response.usage_metadata, 'candidates_token_count', 0),
            "total": getattr(response.usage_metadata, 'total_token_count', 0)
        }

    pricing = CONFIG.get("pricing", {"input_per_1k": 0.0025, "output_per_1k": 0.01})
    cost = round(
        (usage["input"] / 1000) * pricing["input_per_1k"] +
        (usage["output"] / 1000) * pricing["output_per_1k"],
        6
    )

    # Session tracking
    update_session_after_chat(session_id, usage["total"], cost, is_new_conversation)

    output_text = response.output_text

    # Update usage stats in database (OpenAI stores actual messages)
    if db:
        try:
            db.update_conversation_usage(
                conversation_id=conversation_id,
                tokens_input=usage["input"],
                tokens_output=usage["output"],
                cost=cost,
                image_url=s3_urls[0] if s3_urls else None  # Store first URL for backward compat
            )
        except Exception as e:
            logger.error(f"Failed to update usage: {e}")

    # Get message count
    message_count = 1
    if db:
        try:
            conv = db.get_conversation(conversation_id)
            if conv:
                message_count = conv.get("message_count", 1)
        except Exception:
            pass

    return ImageChatResponse(
        text=output_text,
        conversation_id=conversation_id,
        user_id=user_id,
        message_number=message_count,
        usage=usage,
        cost=cost,
        image_urls=s3_urls,
        image_count=len(s3_urls)
    )


@app.post("/chat/file", response_model=FileChatResponse, tags=["Chat"])
async def chat_with_file(
    message: str = Form(...),
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    client: OpenAI = Depends(get_openai)
):
    """
    Send a chat message with a file (PDF, DOCX, TXT) to GPT-4.
    - PDF: Sent directly to OpenAI (native support)
    - DOCX/TXT: Text extracted and sent to OpenAI
    """
    user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"

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
    try:
        message, warnings = sanitize_input(
            message,
            max_length=CONFIG.get("max_input_length", 10000)
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Validate file size
    max_size = CONFIG.get("max_file_size_mb", 20) * 1024 * 1024
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {CONFIG.get('max_file_size_mb', 20)}MB"
        )

    # Determine file type
    filename = file.filename or "document"
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
    mime_type = file.content_type or "application/octet-stream"

    # Upload to S3 (required)
    s3_url = None
    if not storage:
        raise HTTPException(status_code=503, detail="Storage not available")

    try:
        result = storage.upload_bytes(
            data=file_content,
            filename=filename,
            user_id=user_id,
            folder="chat_files",
            content_type=mime_type
        )
        s3_url = result.get("url")
        logger.info(f"File uploaded to S3: {s3_url}")
    except StorageError as e:
        logger.error(f"S3 upload failed: {e}")
        raise HTTPException(status_code=503, detail=f"File upload failed: {e}")

    # Create conversation if needed (with session linking)
    session_id = get_or_create_session(user_id)
    is_new_conversation = False

    if not conversation_id:
        is_new_conversation = True
        conv = client.conversations.create(
            metadata={
                "app": "terminal_chatbot_api",
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "has_files": "true",
                "file_type": file_ext
            }
        )
        conversation_id = conv.id

        if db:
            try:
                db.create_conversation(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    metadata={"has_files": "true", "file_type": file_ext},
                    session_id=session_id
                )
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")

    # Prepare input based on file type
    if file_ext == 'pdf':
        # PDF: Send as file content block with message wrapper (conversation mode)
        file_base64 = base64.b64encode(file_content).decode("utf-8")
        input_content = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": filename,
                        "file_data": f"data:application/pdf;base64,{file_base64}"
                    },
                    {"type": "input_text", "text": message}
                ]
            }
        ]
    elif file_ext in ['docx', 'doc']:
        # DOCX: Extract text using python-docx and send as plain text
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(file_content))
            extracted_text = "\n".join([para.text for para in doc.paragraphs])
            # Send as plain string (like basic chat)
            input_content = f"[Document Content from {filename}]:\n{extracted_text}\n\n[User Question]: {message}"
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise HTTPException(status_code=422, detail=f"Failed to extract DOCX content: {e}")
    elif file_ext in ['txt', 'md', 'json', 'yaml', 'yml', 'py', 'js', 'html', 'css', 'xml', 'csv']:
        # Text files: Read content directly and send as plain text
        try:
            text_content = file_content.decode('utf-8')
            # Send as plain string (like basic chat)
            input_content = f"[File Content from {filename}]:\n{text_content}\n\n[User Question]: {message}"
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="File is not a valid text file")
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type: {file_ext}. Supported: pdf, docx, txt, md, json, yaml, py, js, html, css, xml, csv"
        )

    # Call OpenAI (Stateful)
    try:
        response = client.responses.create(
            model="gpt-4o",
            conversation=conversation_id,
            input=input_content,
            instructions=PROMPTS["system_prompt"],
            max_output_tokens=4096
        )
    except (APIConnectionError, OpenAIRateLimitError, APITimeoutError) as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {type(e).__name__}")
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {str(e)}")

    # Extract usage and calculate cost
    usage = {
        "input": getattr(response.usage, 'input_tokens', 0) if response.usage else 0,
        "output": getattr(response.usage, 'output_tokens', 0) if response.usage else 0,
        "total": getattr(response.usage, 'total_tokens', 0) if response.usage else 0
    }

    pricing = CONFIG.get("pricing", {"input_per_1k": 0.0025, "output_per_1k": 0.01})
    cost = round(
        (usage["input"] / 1000) * pricing["input_per_1k"] +
        (usage["output"] / 1000) * pricing["output_per_1k"],
        6
    )

    # Session tracking
    update_session_after_chat(session_id, usage["total"], cost, is_new_conversation)

    output_text = response.output_text

    # Update usage stats in database
    if db:
        try:
            db.update_conversation_usage(
                conversation_id=conversation_id,
                tokens_input=usage["input"],
                tokens_output=usage["output"],
                cost=cost
            )
        except Exception as e:
            logger.error(f"Failed to update usage: {e}")

    # Get message count
    message_count = 1
    if db:
        try:
            conv = db.get_conversation(conversation_id)
            if conv:
                message_count = conv.get("message_count", 1)
        except Exception:
            pass

    return FileChatResponse(
        text=output_text,
        conversation_id=conversation_id,
        user_id=user_id,
        message_number=message_count,
        usage=usage,
        cost=cost,
        file_url=s3_url,
        file_type=file_ext
    )


# =============================================================================
# Conversation Endpoints
# =============================================================================

@app.post("/conversations", response_model=ConversationResponse, tags=["Conversations"])
async def create_conversation(
    request: ConversationCreate,
    client: OpenAI = Depends(get_openai)
):
    """Create a new conversation."""
    conv = client.conversations.create(
        metadata={
            "app": "terminal_chatbot_api",
            "user_id": request.user_id,
            "created_at": datetime.now().isoformat(),
            **(request.metadata or {})
        }
    )

    if db:
        try:
            db.create_conversation(
                conversation_id=conv.id,
                user_id=request.user_id,
                metadata=request.metadata
            )
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")

    return ConversationResponse(
        id=conv.id,
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
    order: str = Query("asc", pattern="^(asc|desc)$"),
    client: OpenAI = Depends(get_openai)
):
    """Get messages for a conversation from OpenAI Conversations API."""
    try:
        validate_conversation_id(conversation_id)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        items = client.conversations.items.list(
            conversation_id=conversation_id,
            limit=limit,
            order=order
        )

        messages = []
        for item in items.data:
            if item.type == "message":
                content = getattr(item.content[0], 'text', '') if item.content else ''
                messages.append(MessageResponse(
                    id=item.id,
                    role=item.role,
                    content=content,
                    created_at=datetime.now().isoformat(),
                    tokens_input=0,
                    tokens_output=0,
                    cost=0
                ))
        return messages
    except Exception as e:
        logger.error(f"Failed to fetch messages from OpenAI: {e}")
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
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "conversations": "/conversations",
            "files": "/files",
            "health": "/health",
            "docs": "/docs"
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
