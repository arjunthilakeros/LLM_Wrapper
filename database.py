"""
Database Module for Terminal Chatbot
PostgreSQL database with connection pooling and migrations.
"""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2 import pool, sql
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from logger import get_logger
from exceptions import ConfigurationError, ChatbotError

logger = get_logger()

# Default database URL
DEFAULT_DATABASE_URL = os.getenv("DATABASE_URL")


class DatabaseError(ChatbotError):
    """Database-specific errors."""
    pass


class Database:
    """PostgreSQL database manager with connection pooling."""

    _instance: Optional['Database'] = None
    _pool: Optional[pool.ThreadedConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(
        cls,
        database_url: str = None,
        min_connections: int = 1,
        max_connections: int = 10
    ) -> 'Database':
        """Initialize database connection pool."""
        if not POSTGRES_AVAILABLE:
            raise ConfigurationError(
                "PostgreSQL support requires psycopg2. Install with: pip install psycopg2-binary"
            )

        if cls._pool is not None:
            return cls._instance

        url = database_url or DEFAULT_DATABASE_URL
        if not url:
            raise ConfigurationError(
                "DATABASE_URL environment variable is required for database support"
            )

        try:
            cls._pool = pool.ThreadedConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                dsn=url
            )
            logger.info("Database connection pool initialized")

            # Run migrations
            instance = cls()
            instance._run_migrations()
            return instance

        except psycopg2.Error as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise DatabaseError(f"Database connection failed: {e}")

    @classmethod
    def get_instance(cls) -> 'Database':
        """Get database instance."""
        if cls._pool is None:
            cls.initialize()
        return cls._instance

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor):
        """Get a cursor with automatic connection handling."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    def _run_migrations(self):
        """Run database migrations."""
        with self.get_cursor() as cursor:
            # Create conversations table (stores only metadata, OpenAI stores messages)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    session_id VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    total_tokens_input INTEGER DEFAULT 0,
                    total_tokens_output INTEGER DEFAULT 0,
                    total_cost DECIMAL(10, 6) DEFAULT 0,
                    image_urls TEXT[] DEFAULT '{}',
                    tags TEXT[] DEFAULT '{}',
                    title VARCHAR(500),
                    summary TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    is_deleted BOOLEAN DEFAULT FALSE
                )
            """)

            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id VARCHAR(255) PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost DECIMAL(10, 6) DEFAULT 0,
                    conversation_count INTEGER DEFAULT 0,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user_id
                ON conversations(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_last_used
                ON conversations(last_used DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_session_id
                ON conversations(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id
                ON sessions(user_id)
            """)

            logger.info("Database migrations completed")

    # Conversation operations
    def create_conversation(
        self,
        conversation_id: str,
        user_id: str,
        metadata: Dict = None,
        session_id: str = None
    ) -> Dict:
        """Create a new conversation."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations (id, user_id, session_id, metadata)
                VALUES (%s, %s, %s, %s)
                RETURNING id, user_id, created_at, message_count
            """, (conversation_id, user_id, session_id, Json(metadata or {})))
            result = cursor.fetchone()
            logger.info(f"Created conversation: {conversation_id}")
            return dict(result)

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a conversation by ID."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM conversations
                WHERE id = %s AND is_deleted = FALSE
            """, (conversation_id,))
            result = cursor.fetchone()
            return dict(result) if result else None

    def list_conversations(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict]:
        """List conversations for a user."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, user_id, created_at, last_used, message_count
                FROM conversations
                WHERE user_id = %s AND is_deleted = FALSE
                ORDER BY last_used DESC
                LIMIT %s OFFSET %s
            """, (user_id, limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def update_conversation(
        self,
        conversation_id: str,
        message_count: int = None,
        metadata: Dict = None
    ):
        """Update conversation details."""
        with self.get_cursor() as cursor:
            updates = ["last_used = CURRENT_TIMESTAMP"]
            params = []

            if message_count is not None:
                updates.append("message_count = %s")
                params.append(message_count)

            if metadata is not None:
                updates.append("metadata = metadata || %s")
                params.append(Json(metadata))

            params.append(conversation_id)

            cursor.execute(f"""
                UPDATE conversations
                SET {', '.join(updates)}
                WHERE id = %s
            """, params)

    def delete_conversation(self, conversation_id: str, soft: bool = True):
        """Delete a conversation (soft delete by default)."""
        with self.get_cursor() as cursor:
            if soft:
                cursor.execute("""
                    UPDATE conversations SET is_deleted = TRUE WHERE id = %s
                """, (conversation_id,))
            else:
                cursor.execute("""
                    DELETE FROM conversations WHERE id = %s
                """, (conversation_id,))
            logger.info(f"Deleted conversation: {conversation_id}")

    # Usage tracking (OpenAI stores actual messages)
    def update_conversation_usage(
        self,
        conversation_id: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost: float = 0,
        image_url: str = None
    ):
        """Update conversation usage statistics after a chat."""
        with self.get_cursor() as cursor:
            # Base update
            updates = [
                "message_count = message_count + 1",
                "last_used = CURRENT_TIMESTAMP",
                "total_tokens_input = total_tokens_input + %s",
                "total_tokens_output = total_tokens_output + %s",
                "total_cost = total_cost + %s"
            ]
            params = [tokens_input, tokens_output, cost]

            # Add image URL if provided
            if image_url:
                updates.append("image_urls = array_append(image_urls, %s)")
                params.append(image_url)

            params.append(conversation_id)

            cursor.execute(f"""
                UPDATE conversations
                SET {', '.join(updates)}
                WHERE id = %s
            """, params)
            logger.debug(f"Updated usage for conversation {conversation_id}")

    def add_tag(self, conversation_id: str, tag: str):
        """Add a tag to a conversation."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE conversations
                SET tags = array_append(tags, %s)
                WHERE id = %s AND NOT (%s = ANY(tags))
            """, (tag, conversation_id, tag))

    def set_title(self, conversation_id: str, title: str):
        """Set conversation title."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE conversations
                SET title = %s
                WHERE id = %s
            """, (title, conversation_id))

    def set_summary(self, conversation_id: str, summary: str):
        """Set conversation summary."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE conversations
                SET summary = %s
                WHERE id = %s
            """, (summary, conversation_id))

    # Session operations
    def create_session(self, session_id: str, user_id: str) -> Dict:
        """Create a new session."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO sessions (id, user_id)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET last_activity = CURRENT_TIMESTAMP
                RETURNING *
            """, (session_id, user_id))
            return dict(cursor.fetchone())

    def update_session_stats(
        self,
        session_id: str,
        tokens: int,
        cost: float,
        increment_conversations: bool = False
    ):
        """Update session statistics."""
        with self.get_cursor() as cursor:
            if increment_conversations:
                cursor.execute("""
                    UPDATE sessions
                    SET total_tokens = total_tokens + %s,
                        total_cost = total_cost + %s,
                        conversation_count = conversation_count + 1,
                        last_activity = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (tokens, cost, session_id))
            else:
                cursor.execute("""
                    UPDATE sessions
                    SET total_tokens = total_tokens + %s,
                        total_cost = total_cost + %s,
                        last_activity = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (tokens, cost, session_id))

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM sessions WHERE id = %s
            """, (session_id,))
            result = cursor.fetchone()
            return dict(result) if result else None

    def end_session(self, session_id: str):
        """Mark session as inactive."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE sessions SET is_active = FALSE WHERE id = %s
            """, (session_id,))

    # Analytics
    def get_user_stats(self, user_id: str, days: int = 30) -> Dict:
        """Get usage statistics for a user."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_conversations,
                    COALESCE(SUM(total_tokens_input), 0) as total_tokens_input,
                    COALESCE(SUM(total_tokens_output), 0) as total_tokens_output,
                    COALESCE(SUM(total_cost), 0) as total_cost,
                    COALESCE(SUM(message_count), 0) as total_messages
                FROM conversations
                WHERE user_id = %s
                AND created_at > CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND is_deleted = FALSE
            """, (user_id, days))
            return dict(cursor.fetchone())

    # Health check
    def health_check(self) -> Dict:
        """Check database health."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.execute("SELECT COUNT(*) as count FROM conversations")
                conv_count = cursor.fetchone()['count']
            return {
                "status": "healthy",
                "conversations": conv_count,
                "pool_size": self._pool.maxconn if self._pool else 0
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    @classmethod
    def close(cls):
        """Close all connections."""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            cls._instance = None
            logger.info("Database connections closed")


def get_database() -> Database:
    """Get database instance."""
    return Database.get_instance()
