"""
Production-Ready Terminal Chatbot - OpenAI Responses API Version
Uses ONLY the OpenAI Responses API and Conversations API for persistence.
NO local message history storage - all state is managed by OpenAI.

Requires: pip install openai>=1.50.0 pyyaml tenacity psycopg2-binary boto3 python-dotenv
Optional: pip install PyPDF2 python-docx (for PDF/DOCX support)
"""

import atexit
import json
import os
import signal
import sys
import tempfile
import uuid
import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

try:
    import yaml
    from openai import APIConnectionError, RateLimitError, APITimeoutError, APIError
except ImportError as e:
    print(f"Missing package: {e}")
    print("Install: pip install openai>=1.50.0 pyyaml tenacity")
    sys.exit(1)

# Import production modules
from logger import setup_logging, get_logger
from exceptions import (
    ChatbotError,
    ConfigurationError,
    RateLimitExceededError,
    CostLimitExceededError,
    ValidationError,
    APIError as ChatbotAPIError,
    FileProcessingError
)
from validators import sanitize_input, validate_conversation_id, validate_file_path
from rate_limiter import TokenBucketRateLimiter
from config_validator import validate_config

# Import Responses API client ONLY (NO Chat Completions)
from responses_client import (
    ResponsesAPIClient,
    create_responses_client,
    extract_output_text,
    extract_usage,
)

# Import Context Manager for Summary + Window strategy
from context_manager import ContextManager, ContextConfig

# Optional database support
try:
    from database import Database, get_database, POSTGRES_AVAILABLE
    DATABASE_AVAILABLE = POSTGRES_AVAILABLE
except ImportError:
    DATABASE_AVAILABLE = False
    Database = None

# Optional S3 storage support
try:
    from storage import S3Storage, LocalStorage, get_storage, S3_AVAILABLE, S3_ENABLED
    STORAGE_AVAILABLE = S3_AVAILABLE
except ImportError:
    STORAGE_AVAILABLE = False
    S3_ENABLED = False

# Optional health check support
try:
    from health import health_check, print_health_status
    HEALTH_AVAILABLE = True
except ImportError:
    HEALTH_AVAILABLE = False

# Optional imports for document parsing
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# Get base directory
BASE_DIR = Path(__file__).parent

# Supported file types
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
DOCUMENT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.html', '.css', '.xml', '.yaml', '.yml'}
PDF_EXTENSIONS = {'.pdf'}
DOCX_EXTENSIONS = {'.docx'}

# Database enabled flag from environment
DATABASE_ENABLED = os.getenv("DATABASE_ENABLED", "false").lower() == "true"


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = BASE_DIR / "config.yaml"
    default_config = {
        "model": "gpt-4o",
        "max_history_items": 100,
        "rate_limit_per_minute": 10,
        "max_input_length": 10000,
        "max_file_size_mb": 20,

        "warn_at_cost": 1.0,
        "pricing": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
        "data_dir": "./data",
        "export_dir": "./exports",
        "upload_dir": "./uploads",
        "show_tokens": True,
        "show_cost": True,
        "stream_responses": True,
        "logging": {"level": "INFO", "log_to_file": True, "log_dir": "./logs"},
        "api_timeout": 30,
        "api_max_retries": 3,
        "temperature": 1.0,
        "max_tokens": None
    }

    if config_path.exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
            default_config.update(user_config)

    return default_config


def load_prompts() -> dict:
    """Load prompts from prompts.yaml"""
    prompts_path = BASE_DIR / "prompts.yaml"
    default_prompts = {
        "system_prompt": "You are a helpful AI assistant. Be concise and helpful.",
        "welcome_message": "Terminal Chatbot - OpenAI Responses API",
        "custom_prompts": {}
    }

    if prompts_path.exists():
        with open(prompts_path, "r") as f:
            user_prompts = yaml.safe_load(f) or {}
            default_prompts.update(user_prompts)

    return default_prompts


# Load and validate configuration
try:
    CONFIG = validate_config(load_config())
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    sys.exit(1)

PROMPTS = load_prompts()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup logging
logging_config = CONFIG.get("logging", {})
logger = setup_logging(
    level=logging_config.get("level", "INFO"),
    log_to_file=logging_config.get("log_to_file", True),
    log_dir=logging_config.get("log_dir", "./logs")
)

# Initialize database if available and enabled
db = None
if DATABASE_AVAILABLE and DATABASE_ENABLED:
    try:
        db = Database.initialize()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed, using JSON fallback: {e}")
        db = None


class FileHandler:
    """Handle file uploads - images and documents with OpenAI Responses API format"""

    @staticmethod
    def get_file_type(filepath: Path) -> str:
        """Determine file type category"""
        ext = filepath.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        elif ext in DOCUMENT_EXTENSIONS:
            return "document"
        elif ext in PDF_EXTENSIONS:
            return "pdf"
        elif ext in DOCX_EXTENSIONS:
            return "docx"
        else:
            return "unknown"

    @staticmethod
    def validate_file(filepath: Path) -> tuple[bool, str]:
        """Validate file exists and is within size limit"""
        if not filepath.exists():
            return False, f"File not found: {filepath}"

        size_mb = filepath.stat().st_size / (1024 * 1024)
        max_size = CONFIG.get("max_file_size_mb", 20)

        if size_mb > max_size:
            return False, f"File too large: {size_mb:.1f}MB (max {max_size}MB)"

        file_type = FileHandler.get_file_type(filepath)
        if file_type == "unknown":
            return False, f"Unsupported file type: {filepath.suffix}"

        if file_type == "pdf" and not PDF_SUPPORT:
            return False, "PDF support requires: pip install PyPDF2"

        if file_type == "docx" and not DOCX_SUPPORT:
            return False, "DOCX support requires: pip install python-docx"

        return True, "OK"

    @staticmethod
    def read_image(filepath: Path) -> dict:
        """Read image and encode as base64, return Responses API format"""
        mime_type, _ = mimetypes.guess_type(str(filepath))
        if not mime_type:
            mime_type = "image/jpeg"

        with open(filepath, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Return Responses API format: image_url content part
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_data}",
                "detail": "auto"
            },
            "filename": filepath.name,
            "size_kb": filepath.stat().st_size / 1024
        }

    @staticmethod
    def read_document(filepath: Path) -> dict:
        """Read text document, return Responses API format"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                content = f.read()

        # Return Responses API format: text content part
        return {
            "type": "text",
            "text": f"[Document: {filepath.name}]\n```\n{content}\n```",
            "filename": filepath.name,
            "size_kb": filepath.stat().st_size / 1024
        }

    @staticmethod
    def read_pdf(filepath: Path) -> dict:
        """Read PDF document, return Responses API format"""
        if not PDF_SUPPORT:
            return {"type": "error", "message": "PDF support not installed"}

        text_parts = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text())
                text_parts.append("\n")

        content = "".join(text_parts)
        
        # Return Responses API format: text content part
        return {
            "type": "text",
            "text": f"[PDF Document: {filepath.name}]\n```\n{content}\n```",
            "filename": filepath.name,
            "pages": len(reader.pages),
            "size_kb": filepath.stat().st_size / 1024
        }

    @staticmethod
    def read_docx(filepath: Path) -> dict:
        """Read DOCX document, return Responses API format"""
        if not DOCX_SUPPORT:
            return {"type": "error", "message": "DOCX support not installed"}

        doc = docx.Document(filepath)
        content = "\n".join([para.text for para in doc.paragraphs])

        # Return Responses API format: text content part
        return {
            "type": "text",
            "text": f"[DOCX Document: {filepath.name}]\n```\n{content}\n```",
            "filename": filepath.name,
            "size_kb": filepath.stat().st_size / 1024
        }

    @staticmethod
    def process_file(filepath: Path) -> dict:
        """Process any supported file and return Responses API compatible content part"""
        file_type = FileHandler.get_file_type(filepath)

        if file_type == "image":
            return FileHandler.read_image(filepath)
        elif file_type == "document":
            return FileHandler.read_document(filepath)
        elif file_type == "pdf":
            return FileHandler.read_pdf(filepath)
        elif file_type == "docx":
            return FileHandler.read_docx(filepath)
        else:
            return {"type": "error", "message": f"Unsupported: {filepath.suffix}"}


class ConversationManager:
    """
    Production-ready conversation manager using OpenAI Responses API ONLY.
    
    Key characteristics:
    - Uses ONLY responses.create (NO chat.completions.create)
    - Uses ONLY conversation_id from OpenAI Conversations API for persistence
    - NO local message history storage - history comes from OpenAI API
    """

    def __init__(self, user_id: str = None):
        # Initialize Responses API Client (NOT OpenAI client directly)
        self.client = create_responses_client(
            api_key=OPENAI_API_KEY,
            timeout=CONFIG.get("api_timeout", 30),
            max_retries=CONFIG.get("api_max_retries", 3)
        )
        
        self.user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        
        # Local tracking ID (for database/JSON persistence)
        self.conversation_id = None
        
        # OpenAI Conversation ID from Conversations API - THIS IS THE SOURCE OF TRUTH
        self.openai_conversation_id = None
        
        self.message_count = 0
        self.total_tokens = {"input": 0, "output": 0, "total": 0}
        self.total_cost = 0.0
        self.conversations_list = []
        self.pending_files = []
        self._should_shutdown = False
        
        # NO local message history - we use OpenAI's conversation storage
        # self.messages is removed/kept minimal for compatibility only

        # Database reference
        self.db = db
        self.use_database = db is not None

        # Initialize Context Manager for Summary + Window strategy
        self.context_config = ContextConfig.from_dict(CONFIG)
        self.context_manager = ContextManager(
            config=self.context_config,
            responses_client=self.client,
            database=self.db,
            data_dir=BASE_DIR / CONFIG["data_dir"],
        )
        logger.info(f"Context mode: {self.context_config.mode}")

        # Initialize rate limiter
        self.rate_limiter = TokenBucketRateLimiter(
            requests_per_minute=CONFIG.get("rate_limit_per_minute", 10)
        )

        # Create data directories
        self.data_dir = BASE_DIR / CONFIG["data_dir"]
        self.export_dir = BASE_DIR / CONFIG["export_dir"]
        self.upload_dir = BASE_DIR / CONFIG.get("upload_dir", "./uploads")
        self.data_dir.mkdir(exist_ok=True)
        self.export_dir.mkdir(exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)

        # Load conversations
        if self.use_database:
            self._load_conversations_from_db()
        else:
            self._load_conversations_list()

        # Register cleanup handlers
        atexit.register(self._cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        storage_status = "Database" if self.use_database else "JSON files"
        logger.info(f"ConversationManager initialized for user {self.user_id} ({storage_status})")

    @property
    def should_shutdown(self) -> bool:
        return self._should_shutdown

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self._should_shutdown = True
        raise KeyboardInterrupt()

    def _cleanup(self):
        logger.info(
            f"Cleanup - User: {self.user_id}, "
            f"Tokens: {self.total_tokens}, Cost: ${self.total_cost:.4f}"
        )
        self._save_conversations_list()

    def _get_user_file(self) -> Path:
        return self.data_dir / f"{self.user_id}_conversations.json"

    def _load_conversations_from_db(self):
        """Load conversations from database."""
        if self.db:
            try:
                convs = self.db.list_conversations(self.user_id, limit=100)
                self.conversations_list = [
                    {
                        "id": c["id"],
                        "created_at": c["created_at"].isoformat() if hasattr(c["created_at"], 'isoformat') else str(c["created_at"]),
                        "message_count": c["message_count"],
                        "last_used": c.get("last_used")
                    }
                    for c in convs
                ]
                logger.debug(f"Loaded {len(self.conversations_list)} conversations from database")
            except Exception as e:
                logger.error(f"Failed to load conversations from database: {e}")
                self.conversations_list = []

    def _load_conversations_list(self):
        """Load conversations from JSON file (fallback)."""
        user_file = self._get_user_file()
        if user_file.exists():
            try:
                with open(user_file, "r") as f:
                    data = json.load(f)
                    self.conversations_list = data.get("conversations", [])
                logger.debug(f"Loaded {len(self.conversations_list)} conversations from JSON")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse conversations file: {e}", exc_info=True)
                self.conversations_list = []
            except IOError as e:
                logger.error(f"Failed to read conversations file: {e}", exc_info=True)
                self.conversations_list = []

    def _save_conversations_list(self):
        """Save conversations list with atomic write (JSON fallback)."""
        if self.use_database:
            return  # Database handles persistence

        user_file = self._get_user_file()
        data = {
            "user_id": self.user_id,
            "updated_at": datetime.now().isoformat(),
            "conversations": self.conversations_list
        }

        temp_fd = None
        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=str(self.data_dir),
                prefix=".conv_",
                suffix=".tmp"
            )
            with os.fdopen(temp_fd, 'w') as f:
                temp_fd = None
                json.dump(data, f, indent=2)

            os.replace(temp_path, user_file)
            logger.debug(f"Saved conversations list to {user_file}")

        except Exception as e:
            logger.error(f"Failed to save conversations: {e}", exc_info=True)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
        finally:
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass

    def _check_cost_limit(self):
        limit = CONFIG.get("cost_limit", 5.0)
        if self.total_cost >= limit:
            logger.warning(f"Cost limit exceeded: ${self.total_cost:.4f} >= ${limit:.2f}")
            raise CostLimitExceededError(
                message="Cost limit exceeded. Please start a new conversation.",
                current_cost=self.total_cost,
                limit=limit
            )

    def _check_rate_limit(self):
        try:
            self.rate_limiter.acquire(block=False)
        except RateLimitExceededError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            raise

    def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute API operation with retry logic from ResponsesAPIClient."""
        try:
            return operation(*args, **kwargs)
        except (APIConnectionError, RateLimitError, APITimeoutError) as e:
            logger.error(f"API call failed after retries: {e}", exc_info=True)
            raise ChatbotAPIError(
                message=f"API call failed: {type(e).__name__}",
                api_error=str(e),
                retries_attempted=CONFIG.get("api_max_retries", 3)
            ) from e

    def create_conversation(self) -> str:
        """
        Create a new conversation using OpenAI Conversations API.
        The OpenAI conversation_id is the source of truth for persistence.
        """
        # Generate a local conversation ID for tracking
        conv_id = f"conv_{uuid.uuid4().hex[:16]}"
        self.conversation_id = conv_id
        self.message_count = 0

        # Create conversation in OpenAI Conversations API
        try:
            system_prompt = PROMPTS.get("system_prompt")
            conversation = self.client.create_conversation(
                metadata={
                    "user_id": self.user_id,
                    "local_conv_id": conv_id,
                    "app": "terminal_chatbot",
                    "api_version": "responses-api"
                }
            )
            self.openai_conversation_id = conversation.id
            logger.info(f"Created OpenAI conversation: {self.openai_conversation_id}")
        except Exception as e:
            logger.error(f"Failed to create OpenAI conversation: {e}")
            # Continue without OpenAI conversation - will be created on first message
            self.openai_conversation_id = None

        # Save to database or local list
        if self.use_database and self.db:
            try:
                self.db.create_conversation(
                    conversation_id=conv_id,
                    user_id=self.user_id,
                    metadata={
                        "app": "terminal_chatbot",
                        "api_version": "responses-api",
                        "openai_conversation_id": self.openai_conversation_id
                    }
                )
            except Exception as e:
                logger.error(f"Failed to save conversation to database: {e}")

        self.conversations_list.append({
            "id": conv_id,
            "created_at": datetime.now().isoformat(),
            "message_count": 0,
            "openai_conversation_id": self.openai_conversation_id
        })

        if not self.use_database:
            self._save_conversations_list()

        logger.info(f"Created new conversation: {conv_id} (OpenAI: {self.openai_conversation_id})")
        return conv_id

    def upload_file(self, filepath: str) -> dict:
        """Upload a file and store as Responses API compatible content part."""
        try:
            path, warnings = validate_file_path(filepath)
            for warning in warnings:
                logger.warning(f"File path warning: {warning}")
        except ValidationError as e:
            logger.error(f"File path validation failed: {e}")
            return {"success": False, "error": str(e)}

        valid, message = FileHandler.validate_file(path)
        if not valid:
            logger.warning(f"File validation failed: {message}")
            return {"success": False, "error": message}

        file_data = FileHandler.process_file(path)
        if file_data.get("type") == "error":
            logger.error(f"File processing failed: {file_data['message']}")
            return {"success": False, "error": file_data["message"]}

        self.pending_files.append(file_data)
        logger.info(f"File uploaded: {file_data.get('filename', 'unknown')} ({file_data.get('size_kb', 0):.1f}KB)")
        return {
            "success": True,
            "filename": file_data.get("filename", "unknown"),
            "type": file_data.get("type", "unknown"),
            "size_kb": file_data.get("size_kb", 0),
            "pending_count": len(self.pending_files)
        }

    def _build_input(self, user_message: str) -> Union[str, List[Dict[str, Any]]]:
        """
        Build Responses API input format with pending files.
        Returns either a simple string or list of message objects.
        """
        if not self.pending_files:
            # Simple text input
            return user_message

        # Build multimodal content for Responses API
        content_parts = [{"type": "text", "text": user_message}]
        
        for file_data in self.pending_files:
            file_type = file_data.get("type")
            
            if file_type == "image_url":
                # Add image content part
                content_parts.append({
                    "type": "image_url",
                    "image_url": file_data.get("image_url", {})
                })
            elif file_type == "text":
                # Add document text as a separate content part
                content_parts.append({
                    "type": "text",
                    "text": f"\n\n{file_data.get('text', '')}"
                })
        
        # Clear pending files after building input
        self.pending_files = []
        
        # Return as a single user message with content array
        return [{"role": "user", "content": content_parts}]

    def _extract_usage(self, response) -> dict:
        """Extract token usage from Responses API response."""
        usage_data = extract_usage(response)
        if not usage_data:
            return {"input": 0, "output": 0, "total": 0}
        
        # Map Responses API format to internal format
        return {
            "input": usage_data.get("input_tokens", 0),
            "output": usage_data.get("output_tokens", 0),
            "total": usage_data.get("total_tokens", 0)
        }

    def _calculate_cost(self, usage: dict) -> float:
        """Calculate cost based on token usage."""
        pricing = CONFIG["pricing"]
        input_cost = (usage.get("input", 0) / 1000) * pricing["input_per_1k"]
        output_cost = (usage.get("output", 0) / 1000) * pricing["output_per_1k"]
        return round(input_cost + output_cost, 6)

    def _update_usage_stats(self, usage: dict):
        """Update token and cost statistics."""
        self.total_tokens["input"] += usage.get("input", 0)
        self.total_tokens["output"] += usage.get("output", 0)
        self.total_tokens["total"] += usage.get("total", 0)
        cost = self._calculate_cost(usage)
        self.total_cost += cost

        if self.total_cost >= CONFIG.get("warn_at_cost", 1.0):
            print(f"\n[Warning: Total cost ${self.total_cost:.4f}]")

    def _update_conversation_count(self):
        """Update conversation message count."""
        for conv in self.conversations_list:
            if conv["id"] == self.conversation_id:
                conv["message_count"] = self.message_count
                conv["last_used"] = datetime.now().isoformat()
                break

        if self.use_database and self.db:
            try:
                self.db.update_conversation(
                    self.conversation_id,
                    message_count=self.message_count
                )
            except Exception as e:
                logger.error(f"Failed to update conversation: {e}")
        else:
            self._save_conversations_list()

    def _ensure_conversation(self):
        """Ensure we have an OpenAI conversation ID."""
        if not self.conversation_id:
            self.create_conversation()
        
        # Create OpenAI conversation if needed
        if not self.openai_conversation_id:
            try:
                conversation = self.client.create_conversation(
                    metadata={
                        "user_id": self.user_id,
                        "local_conv_id": self.conversation_id,
                        "app": "terminal_chatbot"
                    }
                )
                self.openai_conversation_id = conversation.id
                logger.info(f"Created OpenAI conversation on first message: {self.openai_conversation_id}")
                
                # Update local record with OpenAI conversation ID
                for conv in self.conversations_list:
                    if conv["id"] == self.conversation_id:
                        conv["openai_conversation_id"] = self.openai_conversation_id
                        break
                        
            except Exception as e:
                logger.error(f"Failed to create OpenAI conversation: {e}")
                raise ChatbotAPIError(f"Failed to create conversation: {e}") from e

    def chat(self, message: str) -> dict:
        """
        Send a chat message using ONLY OpenAI Responses API.
        NO chat.completions.create - uses ONLY responses.create.

        Context Management:
        - mode="full": Uses conversation_id, OpenAI handles full history
        - mode="summary_window": Manually builds [Summary] + [Window] context
        """
        self._check_cost_limit()
        self._check_rate_limit()

        try:
            message, warnings = sanitize_input(
                message,
                max_length=CONFIG.get("max_input_length", 10000)
            )
            for warning in warnings:
                logger.warning(f"Input sanitization: {warning}")
                print(f"[Warning: {warning}]")
        except ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            raise

        # Ensure conversation exists
        self._ensure_conversation()

        # Build input with any pending files
        base_input = self._build_input(message)

        # Determine context strategy
        use_summary_window = (
            self.context_config.mode == "summary_window" and
            self.message_count >= self.context_config.summarize_after_messages
        )

        context_metadata = {}

        if use_summary_window:
            # Summary + Window mode: manually build context
            context_result = self.context_manager.build_context_for_api(
                conversation_id=self.openai_conversation_id,
                new_message=message if isinstance(base_input, str) else message,
                system_prompt=PROMPTS.get("system_prompt"),
                message_count=self.message_count
            )

            input_data = context_result["input"]
            use_conversation_id = context_result["use_conversation_id"]
            context_metadata = context_result["metadata"]

            # Log context savings
            logger.info(
                f"Context mode: summary_window, "
                f"tokens: {context_metadata.get('total_tokens', 0)} "
                f"(summary={context_metadata.get('summary_tokens', 0)}, "
                f"window={context_metadata.get('window_tokens', 0)})"
            )
        else:
            # Full mode: let OpenAI handle context
            input_data = base_input
            use_conversation_id = True
            context_metadata = {"mode": "full"}

        # Build API request parameters
        request_params = {
            "model": CONFIG["model"],
            "input": input_data,
            "instructions": PROMPTS.get("system_prompt"),
            "temperature": CONFIG.get("temperature", 1.0),
            "max_output_tokens": CONFIG.get("max_tokens"),
            "user": self.user_id,
            "store": True  # Always store for history retrieval
        }

        # Only include conversation_id if using full mode
        if use_conversation_id:
            request_params["conversation_id"] = self.openai_conversation_id

        # Make Responses API call
        response = self._execute_with_retry(
            self.client.create_response,
            **request_params
        )

        # Extract response text using Responses API helper
        output_text = extract_output_text(response)
        usage = self._extract_usage(response)

        # Update conversation state
        self.message_count += 1
        self._update_conversation_count()
        self._update_usage_stats(usage)

        logger.debug(f"Chat completed - tokens: {usage}, cost: ${self._calculate_cost(usage):.6f}")

        return {
            "text": output_text,
            "usage": usage,
            "cost": self._calculate_cost(usage),
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "openai_conversation_id": self.openai_conversation_id,
            "message_number": self.message_count,
            "context_mode": context_metadata.get("mode", "full"),
            "context_tokens": context_metadata.get("total_tokens", usage.get("input", 0))
        }

    def chat_stream(self, message: str) -> dict:
        """
        Send a chat message with streaming using ONLY OpenAI Responses API.
        Uses streaming from Responses API with event types: response.output_text.delta

        Context Management:
        - mode="full": Uses conversation_id, OpenAI handles full history
        - mode="summary_window": Manually builds [Summary] + [Window] context
        """
        self._check_cost_limit()
        self._check_rate_limit()

        try:
            message, warnings = sanitize_input(
                message,
                max_length=CONFIG.get("max_input_length", 10000)
            )
            for warning in warnings:
                logger.warning(f"Input sanitization: {warning}")
                print(f"[Warning: {warning}]")
        except ValidationError as e:
            logger.error(f"Input validation failed: {e}")
            raise

        # Ensure conversation exists
        self._ensure_conversation()

        # Build input with any pending files
        base_input = self._build_input(message)

        # Determine context strategy
        use_summary_window = (
            self.context_config.mode == "summary_window" and
            self.message_count >= self.context_config.summarize_after_messages
        )

        context_metadata = {}

        if use_summary_window:
            # Summary + Window mode: manually build context
            context_result = self.context_manager.build_context_for_api(
                conversation_id=self.openai_conversation_id,
                new_message=message if isinstance(base_input, str) else message,
                system_prompt=PROMPTS.get("system_prompt"),
                message_count=self.message_count
            )

            input_data = context_result["input"]
            use_conversation_id = context_result["use_conversation_id"]
            context_metadata = context_result["metadata"]

            # Log context savings
            logger.info(
                f"Stream context mode: summary_window, "
                f"tokens: {context_metadata.get('total_tokens', 0)}"
            )
        else:
            # Full mode: let OpenAI handle context
            input_data = base_input
            use_conversation_id = True
            context_metadata = {"mode": "full"}

        # Update message count before streaming
        self.message_count += 1
        self._update_conversation_count()

        # Build streaming request parameters
        stream_params = {
            "model": CONFIG["model"],
            "input": input_data,
            "instructions": PROMPTS.get("system_prompt"),
            "temperature": CONFIG.get("temperature", 1.0),
            "max_output_tokens": CONFIG.get("max_tokens"),
            "user": self.user_id,
            "store": True  # Store the response
        }

        # Only include conversation_id if using full mode
        if use_conversation_id:
            stream_params["conversation_id"] = self.openai_conversation_id

        # Make Responses API streaming call
        stream = self._execute_with_retry(
            self.client.create_response_stream_raw,
            **stream_params
        )

        text_parts = []
        usage = {"input": 0, "output": 0, "total": 0}

        # Process streaming events from Responses API
        for event in stream:
            # Responses API streaming event types
            if hasattr(event, 'type'):
                event_type = event.type
                
                # Handle text delta events
                if event_type == 'response.output_text.delta':
                    if hasattr(event, 'delta') and event.delta:
                        print(event.delta, end='', flush=True)
                        text_parts.append(event.delta)
                
                # Handle completed event for usage stats
                elif event_type == 'response.completed':
                    if hasattr(event, 'response') and event.response:
                        usage = self._extract_usage(event.response)
                        break
            
            # Alternative: handle direct text attributes
            elif hasattr(event, 'delta') and event.delta:
                delta_text = event.delta
                if isinstance(delta_text, str):
                    print(delta_text, end='', flush=True)
                    text_parts.append(delta_text)

        full_text = "".join(text_parts)

        # Update usage stats with actual usage from completed response
        self._update_usage_stats(usage)

        logger.debug(f"Stream completed - tokens: {usage}, cost: ${self._calculate_cost(usage):.6f}")

        return {
            "text": full_text,
            "usage": usage,
            "cost": self._calculate_cost(usage),
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "openai_conversation_id": self.openai_conversation_id,
            "message_number": self.message_count,
            "context_mode": context_metadata.get("mode", "full"),
            "context_tokens": context_metadata.get("total_tokens", usage.get("input", 0))
        }

    def chat_with_image(self, message: str, image_path: str) -> dict:
        """Send a message with an image."""
        result = self.upload_file(image_path)
        if not result["success"]:
            return {"text": f"Upload failed: {result['error']}", "usage": {}, "error": True}
        return self.chat(message)

    def get_history(self) -> list:
        """
        Get conversation history from OpenAI API (NOT local storage).
        Fetches from OpenAI: client.list_conversation_items(conversation_id)
        """
        if not self.openai_conversation_id:
            return []
        
        try:
            # Fetch conversation items from OpenAI Conversations API
            items = self.client.list_conversation_items(
                self.openai_conversation_id,
                limit=CONFIG.get("max_history_items", 100),
                order="asc"
            )
            
            # Convert items to message format
            history = []
            for item in items.data if hasattr(items, 'data') else items:
                role = getattr(item, 'role', None)
                content = getattr(item, 'content', None)
                
                if role and content:
                    # Handle different content types
                    if isinstance(content, list):
                        # Multimodal content - extract text parts
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    text_parts.append(part.get("text", ""))
                                elif part.get("type") == "image_url":
                                    text_parts.append("[Image]")
                        content_text = " ".join(text_parts)
                    elif isinstance(content, str):
                        content_text = content
                    else:
                        content_text = str(content)
                    
                    if role in ["user", "assistant"]:
                        history.append({"role": role, "content": content_text or ""})
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to fetch conversation history from OpenAI: {e}")
            return []

    def list_saved_conversations(self) -> list:
        """List saved conversations."""
        if self.use_database and self.db:
            self._load_conversations_from_db()
        return self.conversations_list

    def load_conversation(self, conv_id: str) -> bool:
        """Load a conversation by ID (supports partial ID matching)."""
        conv_id = conv_id.strip()

        try:
            if self.use_database and self.db:
                # Try exact match first
                conv_data = self.db.get_conversation(conv_id)
                if not conv_data:
                    # Try partial match against user's conversations
                    all_convs = self.db.list_conversations(self.user_id)
                    matches = [c for c in all_convs if c["id"].startswith(conv_id)]
                    if len(matches) == 1:
                        conv_data = matches[0]
                    elif len(matches) > 1:
                        logger.warning(f"Ambiguous conversation ID '{conv_id}', {len(matches)} matches")
                        return False

                if conv_data and conv_data.get("user_id") == self.user_id:
                    self.conversation_id = conv_data["id"]
                    self.message_count = conv_data.get("message_count", 0)
                    # Load OpenAI conversation ID from metadata
                    metadata = conv_data.get("metadata", {})
                    self.openai_conversation_id = metadata.get("openai_conversation_id")

                    logger.info(f"Loaded conversation: {self.conversation_id} (OpenAI: {self.openai_conversation_id})")
                    return True
            else:
                # Check in local conversations list (exact or partial)
                matches = [c for c in self.conversations_list if c["id"] == conv_id or c["id"].startswith(conv_id)]
                if len(matches) == 1:
                    conv = matches[0]
                    self.conversation_id = conv["id"]
                    self.message_count = conv.get("message_count", 0)
                    self.openai_conversation_id = conv.get("openai_conversation_id")

                    logger.info(f"Loaded conversation: {self.conversation_id} (OpenAI: {self.openai_conversation_id})")
                    return True
                elif len(matches) > 1:
                    logger.warning(f"Ambiguous conversation ID '{conv_id}', {len(matches)} matches")
                    return False

            logger.warning(f"Conversation not found: {conv_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}", exc_info=True)
            return False

    def export_conversation(self, format: str = "txt") -> str:
        """Export conversation to a file."""
        history = self.get_history()
        if not history:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{self.conversation_id[:8]}_{timestamp}.{format}"
        filepath = self.export_dir / filename

        temp_fd = None
        temp_path = None
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=str(self.export_dir),
                prefix=".export_",
                suffix=f".{format}.tmp"
            )

            if format == "txt":
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    temp_fd = None
                    f.write(f"Conversation Export\nID: {self.conversation_id}\n")
                    f.write(f"User: {self.user_id}\nDate: {datetime.now().isoformat()}\n")
                    f.write("=" * 50 + "\n\n")
                    for msg in history:
                        role = "You" if msg["role"] == "user" else "AI"
                        f.write(f"{role}:\n{msg['content']}\n\n")

            elif format == "json":
                export_data = {
                    "conversation_id": self.conversation_id,
                    "openai_conversation_id": self.openai_conversation_id,
                    "user_id": self.user_id,
                    "exported_at": datetime.now().isoformat(),
                    "messages": history,
                    "api_version": "responses-api"
                }
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    temp_fd = None
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

            elif format == "md":
                with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                    temp_fd = None
                    f.write(f"# Conversation Export\n\n")
                    f.write(f"- **ID:** {self.conversation_id}\n")
                    f.write(f"- **OpenAI ID:** {self.openai_conversation_id}\n")
                    f.write(f"- **User:** {self.user_id}\n")
                    f.write(f"- **API:** OpenAI Responses API\n\n---\n\n")
                    for msg in history:
                        role = "**You**" if msg["role"] == "user" else "**AI**"
                        f.write(f"{role}:\n\n{msg['content']}\n\n---\n\n")

            os.replace(temp_path, filepath)
            logger.info(f"Exported conversation to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            return None
        finally:
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass

    def get_user_stats(self) -> dict:
        """Get user statistics from database."""
        if self.use_database and self.db:
            try:
                return self.db.get_user_stats(self.user_id)
            except Exception as e:
                logger.error(f"Failed to get user stats: {e}")
        return {
            "total_conversations": len(self.conversations_list),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost
        }


def print_usage(result: dict):
    """Print usage statistics."""
    u = result.get("usage", {})
    cost = result.get("cost", 0)

    print(f"\n{'─' * 55}")
    if CONFIG["show_tokens"]:
        print(f"Tokens: {u.get('input', 0)} in | {u.get('output', 0)} out | {u.get('total', 0)} total")

    # Show context mode info
    context_mode = result.get("context_mode", "full")
    context_tokens = result.get("context_tokens", 0)
    if context_mode == "summary_window":
        print(f"Context: summary_window ({context_tokens} tokens)")
    else:
        print(f"Context: full")

    if CONFIG["show_cost"]:
        print(f"Cost: ${cost:.6f}")
    print(f"User: {result['user_id']} | Conv: {result['conversation_id'][:16]}...")
    print(f"Message: #{result['message_number']}")
    print(f"{'─' * 55}\n")


def _parse_path_and_message(text: str) -> tuple:
    """Parse a quoted or unquoted file path and optional message from input text."""
    text = text.strip()
    if text.startswith('"'):
        end = text.find('"', 1)
        if end != -1:
            return text[1:end], text[end+1:].strip()
        return text.strip('"'), ""
    if text.startswith("'"):
        end = text.find("'", 1)
        if end != -1:
            return text[1:end], text[end+1:].strip()
        return text.strip("'"), ""
    parts = text.split(maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def main():
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set")
        print("Set OPENAI_API_KEY first!")
        return

    user_id = os.getenv("USER_ID")

    print("\n" + "=" * 60)
    print(f"  {PROMPTS['welcome_message']}")
    print("=" * 60)
    print("\nCommands:")
    print("  /new              - New conversation")
    print("  /id               - Show IDs")
    print("  /history          - Show history")
    print("  /tokens           - Show usage statistics")
    print("  /stats            - Show user statistics")
    print("  /upload PATH      - Upload file (image/doc)")
    print("  /image PATH MSG   - Send image with message")
    print("  /load ID          - Load conversation")
    print("  /list             - List saved conversations")
    print("  /export [fmt]     - Export (txt/json/md)")
    print("  /prompt NAME      - Switch prompt preset")
    print("  /config           - Show config")
    print("  /health           - Show system health")
    print("  /quit             - Exit")
    print("=" * 60)

    # Show supported file types
    print(f"\nSupported files:")
    print(f"  Images: {', '.join(IMAGE_EXTENSIONS)}")
    print(f"  Docs:   {', '.join(DOCUMENT_EXTENSIONS)}")
    if PDF_SUPPORT:
        print(f"  PDF:    .pdf")
    if DOCX_SUPPORT:
        print(f"  DOCX:   .docx")

    # Show storage status
    storage_type = "PostgreSQL" if (DATABASE_AVAILABLE and DATABASE_ENABLED and db) else "JSON files"
    print(f"\nStorage: {storage_type}")
    print(f"API: OpenAI Responses API (with Conversations API persistence)")
    print()

    manager = ConversationManager(user_id=user_id)
    print(f"User ID: {manager.user_id}")
    print(f"Conversation: {manager.create_conversation()}")
    print(f"Model: {CONFIG['model']}\n")

    while not manager.should_shutdown:
        try:
            if manager.pending_files:
                print(f"[Pending Files: {len(manager.pending_files)}]")
                for i, pf in enumerate(manager.pending_files, 1):
                    print(f"  {i}. {pf.get('filename', 'unknown')} ({pf.get('type', 'unknown')})")

            user_input = input("You: ").strip()
            if not user_input:
                continue

            if user_input == "/quit":
                print(f"\nTokens: {manager.total_tokens}, Cost: ${manager.total_cost:.6f}")
                print("Goodbye!")
                break

            elif user_input == "/new":
                manager.pending_files = []
                print(f"\nNew: {manager.create_conversation()}\n")

            elif user_input == "/id":
                print(f"\nUser: {manager.user_id}")
                print(f"Conv: {manager.conversation_id}")
                print(f"OpenAI Conv: {manager.openai_conversation_id}")
                print(f"Msgs: {manager.message_count}\n")

            elif user_input == "/tokens":
                t = manager.total_tokens
                print(f"\nTokens: {t['input']:,} in | {t['output']:,} out | {t['total']:,} total")
                print(f"Cost: ${manager.total_cost:.6f}\n")

            elif user_input == "/stats":
                stats = manager.get_user_stats()
                print("\n--- User Statistics ---")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: ${value:.4f}")
                    else:
                        print(f"  {key}: {value}")
                print("-----------------------\n")

            elif user_input == "/history":
                print("\n--- History ---")
                for m in manager.get_history():
                    role = "You" if m["role"] == "user" else "AI"
                    content = m["content"][:80] + "..." if len(m["content"]) > 80 else m["content"]
                    print(f"{role}: {content}")
                print("---------------\n")

            elif user_input == "/list":
                convs = manager.list_saved_conversations()
                print(f"\nConversations ({len(convs)}):")
                for c in convs[-10:]:
                    print(f"  {c['id']} ({c.get('message_count', '?')} msgs)")
                print()

            elif user_input.startswith("/load "):
                conv_id = user_input[6:].strip()
                if manager.load_conversation(conv_id):
                    print(f"\nLoaded: {conv_id}\n")
                else:
                    print(f"\nFailed: {conv_id}\n")

            elif user_input.startswith("/upload "):
                filepath, _ = _parse_path_and_message(user_input[8:].strip())
                result = manager.upload_file(filepath)
                if result["success"]:
                    print(f"\nUploaded: {result['filename']} ({result['size_kb']:.1f}KB)")
                    print("File will be included with your next message.\n")
                else:
                    print(f"\nError: {result['error']}\n")

            elif user_input.startswith("/image "):
                filepath, message = _parse_path_and_message(user_input[7:].strip())
                if not message:
                    print("\nUsage: /image <path> <message>\n")
                    continue
                result = manager.upload_file(filepath)
                if not result["success"]:
                    print(f"\nError: {result['error']}\n")
                    continue

                print(f"[Analyzing: {result['filename']}]")
                print("AI: ", end="", flush=True)
                if CONFIG["stream_responses"]:
                    chat_result = manager.chat_stream(message)
                    print()
                else:
                    chat_result = manager.chat(message)
                    print(chat_result["text"])
                print_usage(chat_result)

            elif user_input.startswith("/export"):
                parts = user_input.split()
                fmt = parts[1] if len(parts) > 1 else "txt"
                if fmt not in ["txt", "json", "md"]:
                    print("Formats: txt, json, md\n")
                else:
                    filepath = manager.export_conversation(fmt)
                    print(f"\nExported: {filepath}\n" if filepath else "\nNo history.\n")

            elif user_input.startswith("/prompt "):
                name = user_input[8:].strip()
                custom = PROMPTS.get("custom_prompts", {})
                if name in custom:
                    PROMPTS["system_prompt"] = custom[name]
                    print(f"\nSwitched to '{name}' prompt.\n")
                else:
                    print(f"\nAvailable: {', '.join(custom.keys())}\n")

            elif user_input == "/config":
                print("\nConfig:")
                for k, v in CONFIG.items():
                    if k != "pricing":
                        print(f"  {k}: {v}")
                print()

            elif user_input == "/health":
                if HEALTH_AVAILABLE:
                    print_health_status()
                else:
                    print("\nHealth check not available (missing psutil)\n")

            elif user_input.startswith("/"):
                print("Unknown command.\n")

            else:
                print("AI: ", end="", flush=True)
                if CONFIG["stream_responses"]:
                    result = manager.chat_stream(user_input)
                    print()
                else:
                    result = manager.chat(user_input)
                    print(result["text"])
                print_usage(result)

        except KeyboardInterrupt:
            print(f"\n\nTokens: {manager.total_tokens}, Cost: ${manager.total_cost:.6f}")
            print("Goodbye!")
            break
        except CostLimitExceededError as e:
            logger.warning(f"Cost limit exceeded: {e}")
            print(f"\nError: {e}\n")
        except RateLimitExceededError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            print(f"\nError: {e}\n")
            if e.retry_after:
                print(f"Please wait {e.retry_after:.1f} seconds before trying again.\n")
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            print(f"\nValidation error: {e}\n")
        except ChatbotAPIError as e:
            logger.error(f"API error: {e}", exc_info=True)
            print(f"\nAPI error: {e}\n")
        except ChatbotError as e:
            logger.error(f"Chatbot error: {e}", exc_info=True)
            print(f"\nError: {e}\n")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
