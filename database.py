"""
Database Module - OpenAI Conversations API Integration
PostgreSQL database for metadata and usage tracking ONLY.

IMPORTANT: NO message content is stored locally.
OpenAI stores ALL conversation history via Conversations API.
This database tracks: conversation metadata, title, summary, usage stats.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from decimal import Decimal

try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

from logger import get_logger
from exceptions import ConfigurationError, ChatbotError

logger = get_logger()

DEFAULT_DATABASE_URL = os.getenv("DATABASE_URL")


class DatabaseError(ChatbotError):
    """Database-specific errors."""
    pass


class Database:
    """
    PostgreSQL database manager.

    Design Principle: OpenAI stores messages, we store metadata.

    What we store:
    - Conversation ID (same as OpenAI conversation ID)
    - Title and summary for UI display
    - Message count for summary triggers
    - Usage statistics (tokens, cost)

    What we DON'T store:
    - Message content (OpenAI handles this)
    - Conversation history (fetch from OpenAI API)
    """

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
                "PostgreSQL requires psycopg2. Install: pip install psycopg2-binary"
            )

        if cls._pool is not None:
            return cls._instance

        url = database_url or DEFAULT_DATABASE_URL
        if not url:
            raise ConfigurationError("DATABASE_URL environment variable required")

        try:
            cls._pool = pool.ThreadedConnectionPool(
                minconn=min_connections,
                maxconn=max_connections,
                dsn=url
            )
            logger.info("Database connection pool initialized")

            instance = cls()
            instance._run_migrations()
            return instance

        except psycopg2.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise DatabaseError(f"Database connection failed: {e}")

    @classmethod
    def get_instance(cls) -> 'Database':
        """Get database instance."""
        if cls._pool is None:
            cls.initialize()
        return cls._instance

    @contextmanager
    def get_connection(self):
        """Get connection from pool."""
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit()
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor):
        """Get cursor with automatic connection handling."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    def _run_migrations(self):
        """Create tables for conversation metadata (NOT message content)."""
        with self.get_cursor() as cursor:
            # Check if old schema exists and migrate
            cursor.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'conversations'
            """)
            existing_columns = {row['column_name'] for row in cursor.fetchall()}

            if not existing_columns:
                # Fresh install - create new schema
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id VARCHAR(255) PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        title VARCHAR(500),
                        summary TEXT,
                        message_count INTEGER DEFAULT 0,
                        summary_message_count INTEGER DEFAULT 0,
                        summary_updated_at TIMESTAMP,
                        total_tokens_input INTEGER DEFAULT 0,
                        total_tokens_output INTEGER DEFAULT 0,
                        total_cost DECIMAL(10, 6) DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_deleted BOOLEAN DEFAULT FALSE,
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                """)
            else:
                # Existing table - apply migrations
                # Add new columns if they don't exist
                if 'title' not in existing_columns:
                    cursor.execute("ALTER TABLE conversations ADD COLUMN title VARCHAR(500)")
                    logger.info("Added 'title' column")

                if 'summary' not in existing_columns:
                    cursor.execute("ALTER TABLE conversations ADD COLUMN summary TEXT")
                    logger.info("Added 'summary' column")

                if 'message_count' not in existing_columns:
                    cursor.execute("ALTER TABLE conversations ADD COLUMN message_count INTEGER DEFAULT 0")
                    logger.info("Added 'message_count' column")

                if 'summary_message_count' not in existing_columns:
                    cursor.execute("ALTER TABLE conversations ADD COLUMN summary_message_count INTEGER DEFAULT 0")
                    logger.info("Added 'summary_message_count' column")

                if 'summary_updated_at' not in existing_columns:
                    cursor.execute("ALTER TABLE conversations ADD COLUMN summary_updated_at TIMESTAMP")
                    logger.info("Added 'summary_updated_at' column")

                # Rename last_used to updated_at if needed
                if 'last_used' in existing_columns and 'updated_at' not in existing_columns:
                    cursor.execute("ALTER TABLE conversations RENAME COLUMN last_used TO updated_at")
                    logger.info("Renamed 'last_used' to 'updated_at'")
                elif 'updated_at' not in existing_columns:
                    cursor.execute("ALTER TABLE conversations ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
                    logger.info("Added 'updated_at' column")

                # Migrate existing summary from metadata to dedicated column
                cursor.execute("""
                    UPDATE conversations
                    SET summary = metadata->'context_summary'->>'summary',
                        summary_message_count = COALESCE((metadata->'context_summary'->>'message_count')::INTEGER, 0)
                    WHERE metadata->'context_summary' IS NOT NULL
                    AND summary IS NULL
                """)

                # Drop redundant openai_conversation_id column if it exists
                if 'openai_conversation_id' in existing_columns:
                    cursor.execute("ALTER TABLE conversations DROP COLUMN IF EXISTS openai_conversation_id")
                    logger.info("Dropped redundant 'openai_conversation_id' column")

            # Create/update indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user_id
                ON conversations(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
                ON conversations(updated_at DESC)
            """)
            # Drop old index if exists
            cursor.execute("""
                DROP INDEX IF EXISTS idx_conversations_openai_id
            """)
            cursor.execute("""
                DROP INDEX IF EXISTS idx_conversations_last_used
            """)
            logger.info("Database migrations completed")

    # =========================================================================
    # Conversation Management
    # =========================================================================

    def create_conversation(
        self,
        conversation_id: str,
        user_id: str,
        title: str = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Create local record for an OpenAI conversation.

        Args:
            conversation_id: OpenAI conversation ID (conv_xxx)
            user_id: Local user ID
            title: Optional conversation title
            metadata: Optional metadata

        Returns:
            Created conversation record
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations (id, user_id, title, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    updated_at = CURRENT_TIMESTAMP,
                    is_deleted = FALSE
                RETURNING id, user_id, title, summary, message_count, created_at, updated_at, metadata
            """, (conversation_id, user_id, title, Json(metadata or {})))
            result = cursor.fetchone()
            logger.info(f"Created conversation: {conversation_id} for user {user_id}")
            return dict(result)

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation by ID."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, user_id, title, summary, message_count,
                       summary_message_count, summary_updated_at,
                       total_tokens_input, total_tokens_output, total_cost,
                       created_at, updated_at, metadata, is_deleted
                FROM conversations
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
        """
        List conversations for a user (metadata only).

        To get actual messages, use OpenAI's conversations.items.list()
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, user_id, title, summary, message_count,
                       total_tokens_input, total_tokens_output, total_cost,
                       created_at, updated_at, metadata
                FROM conversations
                WHERE user_id = %s AND is_deleted = FALSE
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
            """, (user_id, limit, offset))
            return [dict(row) for row in cursor.fetchall()]

    def update_conversation(
        self,
        conversation_id: str,
        title: str = None,
        metadata: Dict = None,
        increment_message_count: bool = False
    ) -> bool:
        """
        Update conversation metadata.

        Args:
            conversation_id: Conversation ID
            title: New title (optional)
            metadata: Metadata to merge (optional)
            increment_message_count: If True, increment message_count by 1

        Returns:
            True if updated, False if not found
        """
        with self.get_cursor() as cursor:
            updates = ["updated_at = CURRENT_TIMESTAMP"]
            params = []

            if title is not None:
                updates.append("title = %s")
                params.append(title)

            if metadata is not None:
                updates.append("metadata = metadata || %s")
                params.append(Json(metadata))

            if increment_message_count:
                updates.append("message_count = message_count + 1")

            params.append(conversation_id)

            cursor.execute(f"""
                UPDATE conversations
                SET {', '.join(updates)}
                WHERE id = %s AND is_deleted = FALSE
                RETURNING id
            """, params)
            result = cursor.fetchone()
            return result is not None

    def delete_conversation(self, conversation_id: str, soft: bool = True):
        """Delete conversation (soft delete by default)."""
        with self.get_cursor() as cursor:
            if soft:
                cursor.execute("""
                    UPDATE conversations SET is_deleted = TRUE
                    WHERE id = %s
                """, (conversation_id,))
            else:
                cursor.execute("""
                    DELETE FROM conversations
                    WHERE id = %s
                """, (conversation_id,))
            logger.info(f"Deleted conversation: {conversation_id}")

    # =========================================================================
    # Usage Tracking
    # =========================================================================

    def update_conversation_usage(
        self,
        conversation_id: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost: float = 0,
        increment_message_count: bool = True
    ) -> Optional[Dict]:
        """
        Update usage stats after each API call.

        Call this after every responses.create() to track costs.

        Args:
            conversation_id: Conversation ID
            tokens_input: Input tokens used
            tokens_output: Output tokens used
            cost: Cost in USD
            increment_message_count: If True, increment message_count by 1

        Returns:
            Updated conversation data with message_count, or None if not found
        """
        with self.get_cursor() as cursor:
            message_increment = "message_count = message_count + 1," if increment_message_count else ""
            cursor.execute(f"""
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP,
                    {message_increment}
                    total_tokens_input = total_tokens_input + %s,
                    total_tokens_output = total_tokens_output + %s,
                    total_cost = total_cost + %s
                WHERE id = %s AND is_deleted = FALSE
                RETURNING id, message_count, summary_message_count
            """, (tokens_input, tokens_output, cost, conversation_id))
            result = cursor.fetchone()
            if result:
                logger.debug(f"Updated usage: {conversation_id} +{tokens_input}in +{tokens_output}out, messages={result['message_count']}")
                return dict(result)
            return None

    # =========================================================================
    # Summary Management (for Summary + Window strategy)
    # =========================================================================

    def update_summary(
        self,
        conversation_id: str,
        summary_text: str,
        message_count: int
    ) -> bool:
        """
        Store or update summary for a conversation.

        Uses dedicated summary column (not metadata JSONB).

        Args:
            conversation_id: Conversation ID
            summary_text: The summary text
            message_count: Current message count when summary was generated

        Returns:
            True if updated, False if not found
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE conversations
                SET summary = %s,
                    summary_message_count = %s,
                    summary_updated_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND is_deleted = FALSE
                RETURNING id
            """, (summary_text, message_count, conversation_id))
            result = cursor.fetchone()
            if result:
                logger.debug(f"Updated summary for {conversation_id} at message {message_count}")
                return True
            return False

    def get_summary(self, conversation_id: str) -> Optional[Dict]:
        """
        Get stored summary for a conversation.

        Returns:
            Dict with 'summary', 'message_count', 'updated_at' or None
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT summary, summary_message_count as message_count, summary_updated_at as updated_at
                FROM conversations
                WHERE id = %s AND is_deleted = FALSE
            """, (conversation_id,))
            result = cursor.fetchone()
            if result and result.get("summary"):
                return dict(result)
            return None

    def update_title(self, conversation_id: str, title: str) -> bool:
        """
        Update conversation title.

        Args:
            conversation_id: Conversation ID
            title: New title

        Returns:
            True if updated, False if not found
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE conversations
                SET title = %s, updated_at = CURRENT_TIMESTAMP
                WHERE id = %s AND is_deleted = FALSE
                RETURNING id
            """, (title, conversation_id))
            result = cursor.fetchone()
            if result:
                logger.debug(f"Updated title for {conversation_id}: {title}")
                return True
            return False

    def should_update_summary(self, conversation_id: str, interval: int = 10) -> bool:
        """
        Check if summary needs to be updated.

        Args:
            conversation_id: Conversation ID
            interval: Number of messages between summary updates

        Returns:
            True if (message_count - summary_message_count) >= interval
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT message_count, summary_message_count
                FROM conversations
                WHERE id = %s AND is_deleted = FALSE
            """, (conversation_id,))
            result = cursor.fetchone()
            if not result:
                return False
            msg_count = result.get('message_count', 0) or 0
            summary_count = result.get('summary_message_count', 0) or 0
            return (msg_count - summary_count) >= interval

    # Backward compatibility aliases
    def update_context_summary(self, conversation_id: str, summary_text: str, message_count: int) -> bool:
        """Deprecated: Use update_summary() instead."""
        return self.update_summary(conversation_id, summary_text, message_count)

    def get_context_summary(self, conversation_id: str) -> Optional[Dict]:
        """Deprecated: Use get_summary() instead."""
        return self.get_summary(conversation_id)

    # =========================================================================
    # Analytics
    # =========================================================================

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

    # =========================================================================
    # Health Check
    # =========================================================================

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
            return {"status": "unhealthy", "error": str(e)}

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
