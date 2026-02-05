"""
Context Manager for Summary + Window Strategy
Reduces API costs by ~75-97% by sending [Summary] + [Last N turns] instead of full history.

Strategy Overview:
- OpenAI Conversations API still STORES all messages (full history retrievable)
- But we MANUALLY build context for SENDING (cost control)
- When mode: summary_window, do NOT use conversation parameter
- Use store=True to persist responses without auto-including history

Token Budget (2,500 total):
- System prompt: ~200 tokens
- Summary: ≤500 tokens
- Window (10 exchanges): ~1,500 tokens
- New message: ~200 tokens
- Buffer: ~100 tokens
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from logger import get_logger

logger = get_logger()

# Optional tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed. Using approximate token counting.")


@dataclass
class ContextConfig:
    """Configuration for context management."""
    mode: str = "full"  # "full" or "summary_window"
    window_size: int = 10  # Last N message pairs to keep in window
    max_context_tokens: int = 2500  # Total token budget for context
    summarize_after_messages: int = 20  # When to start summarizing
    summary_model: str = "gpt-4o-mini"  # Cheap model for summaries
    max_summary_tokens: int = 500  # Maximum tokens for summary
    summary_update_interval: int = 10  # Re-summarize every N messages

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ContextConfig":
        """Create from config dictionary."""
        ctx_config = config.get("context_management", {})
        return cls(
            mode=ctx_config.get("mode", "full"),
            window_size=ctx_config.get("window_size", 10),
            max_context_tokens=ctx_config.get("max_context_tokens", 2500),
            summarize_after_messages=ctx_config.get("summarize_after_messages", 20),
            summary_model=ctx_config.get("summary_model", "gpt-4o-mini"),
            max_summary_tokens=ctx_config.get("max_summary_tokens", 500),
            summary_update_interval=ctx_config.get("summary_update_interval", 10),
        )


class ContextManager:
    """
    Manages conversation context using Summary + Window strategy.

    Key principle:
    - OpenAI still stores all messages via Conversations API
    - But we BUILD context manually to reduce tokens sent
    - Summary captures key info from older messages
    - Window keeps recent messages for immediate context
    """

    # Summary generation prompt
    SUMMARY_PROMPT = """You are a conversation summarizer. Create a concise summary of the conversation history.

Focus on:
1. User's name and any personal information shared
2. Key topics discussed and decisions made
3. Important facts, preferences, or context mentioned
4. Any pending tasks or follow-up items

Be factual and concise. Use bullet points. Keep under 500 tokens.

Conversation to summarize:
{conversation}

Summary:"""

    def __init__(
        self,
        config: ContextConfig,
        responses_client: Any,
        database: Any = None,
        data_dir: Path = None,
    ):
        """
        Initialize ContextManager.

        Args:
            config: ContextConfig instance
            responses_client: ResponsesAPIClient for API calls
            database: Optional database instance for storing summaries
            data_dir: Directory for local JSON fallback storage
        """
        self.config = config
        self.client = responses_client
        self.db = database
        self.data_dir = data_dir or Path("./data")

        # Initialize tiktoken encoder if available
        self._encoder = None
        if TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.encoding_for_model("gpt-4o")
            except Exception:
                try:
                    self._encoder = tiktoken.get_encoding("cl100k_base")
                except Exception as e:
                    logger.warning(f"Failed to initialize tiktoken encoder: {e}")

        logger.info(f"ContextManager initialized: mode={config.mode}, window={config.window_size}")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Uses tiktoken for accurate counting, falls back to estimation.
        """
        if not text:
            return 0

        if self._encoder:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass

        # Fallback: estimate ~4 characters per token
        return len(text) // 4

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count total tokens in a list of messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            # Add overhead for role and formatting (~4 tokens per message)
            total += self.count_tokens(content) + 4
        return total

    def should_use_summary_window(self, message_count: int) -> bool:
        """Check if summary_window mode should be used."""
        if self.config.mode != "summary_window":
            return False
        return message_count >= self.config.summarize_after_messages

    def _fetch_history_from_openai(
        self,
        conversation_id: str,
        limit: int = 100,
    ) -> List[Dict[str, str]]:
        """
        Fetch conversation history from OpenAI Conversations API.

        Returns list of {"role": "user"|"assistant", "content": "..."}
        """
        try:
            items_response = self.client.list_conversation_items(
                conversation_id=conversation_id,
                limit=limit,
                order="asc"
            )

            items = []
            if hasattr(items_response, "data"):
                items = items_response.data
            elif isinstance(items_response, list):
                items = items_response

            messages = []
            for item in items:
                item_type = getattr(item, "type", "")
                if item_type != "message":
                    continue

                role = getattr(item, "role", "")
                if role not in ("user", "assistant"):
                    continue

                content_parts = getattr(item, "content", [])
                text_parts = []

                for part in content_parts:
                    if hasattr(part, "type"):
                        part_type = part.type
                        if part_type in ("input_text", "output_text", "text"):
                            text_parts.append(getattr(part, "text", ""))
                        elif part_type == "input_image":
                            text_parts.append("[Image]")
                        elif part_type == "input_file":
                            text_parts.append(f"[File: {getattr(part, 'filename', 'document')}]")

                if text_parts:
                    messages.append({
                        "role": role,
                        "content": "\n".join(text_parts)
                    })

            return messages

        except Exception as e:
            logger.error(f"Failed to fetch history from OpenAI: {e}")
            return []

    def _get_window_messages(
        self,
        messages: List[Dict[str, str]],
        window_size: int,
    ) -> List[Dict[str, str]]:
        """
        Get the last N message pairs (user + assistant turns).

        Args:
            messages: Full message history
            window_size: Number of pairs to keep

        Returns:
            Last window_size * 2 messages (or all if fewer)
        """
        # Each "turn" is typically a user message + assistant response
        messages_to_keep = window_size * 2

        if len(messages) <= messages_to_keep:
            return messages

        return messages[-messages_to_keep:]

    def _get_messages_to_summarize(
        self,
        messages: List[Dict[str, str]],
        window_size: int,
    ) -> List[Dict[str, str]]:
        """Get messages that should be summarized (everything except window)."""
        messages_to_keep = window_size * 2

        if len(messages) <= messages_to_keep:
            return []

        return messages[:-messages_to_keep]

    def _format_messages_for_summary(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a string for summarization."""
        parts = []
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    async def _generate_summary_async(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """Generate summary using the summary model (async version)."""
        return self._generate_summary(messages)

    def _generate_summary(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Generate summary of messages using a cheap model.

        Uses gpt-4o-mini by default for cost efficiency.
        """
        if not messages:
            return ""

        conversation_text = self._format_messages_for_summary(messages)
        prompt = self.SUMMARY_PROMPT.format(conversation=conversation_text)

        try:
            # Use the responses client to generate summary
            # Don't use conversation_id - we want a standalone summary
            response = self.client.create_response(
                model=self.config.summary_model,
                input=prompt,
                max_output_tokens=self.config.max_summary_tokens,
                temperature=0.3,  # Lower temperature for factual summary
                store=False,  # Don't store the summary generation
            )

            # Extract text from response
            from responses_client import extract_output_text
            summary = extract_output_text(response)

            logger.info(f"Generated summary: {self.count_tokens(summary)} tokens")
            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Fallback: create a basic summary from first/last messages
            return self._create_fallback_summary(messages)

    def _create_fallback_summary(self, messages: List[Dict[str, str]]) -> str:
        """Create a basic fallback summary if API call fails."""
        if not messages:
            return ""

        parts = ["[Conversation Summary - Fallback]"]

        # Include first few exchanges
        for msg in messages[:4]:
            role = "User" if msg["role"] == "user" else "AI"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            parts.append(f"- {role}: {content}")

        if len(messages) > 4:
            parts.append(f"- ... ({len(messages) - 4} more messages)")

        return "\n".join(parts)

    def _get_stored_summary(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored summary for a conversation.

        Uses dedicated summary column in database (preferred),
        falls back to local JSON file if database unavailable.
        """
        # Try database (uses dedicated summary column)
        if self.db:
            try:
                summary_data = self.db.get_summary(conversation_id)
                if summary_data:
                    return summary_data
            except Exception as e:
                logger.debug(f"Database summary lookup failed: {e}")

        # Fallback to local JSON
        summary_file = self.data_dir / f"summaries/{conversation_id}.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Local summary load failed: {e}")

        return None

    def _store_summary(
        self,
        conversation_id: str,
        summary_text: str,
        message_count: int,
    ):
        """
        Store summary for a conversation.

        Uses dedicated summary column in database (preferred),
        falls back to local JSON if database unavailable.
        """
        # Try database (uses dedicated summary column)
        if self.db:
            try:
                self.db.update_summary(
                    conversation_id,
                    summary_text,
                    message_count
                )
                logger.debug(f"Stored summary in database for {conversation_id}")
                return
            except Exception as e:
                logger.debug(f"Database summary storage failed: {e}")

        # Fallback to local JSON
        summary_dir = self.data_dir / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_file = summary_dir / f"{conversation_id}.json"

        summary_data = {
            "summary": summary_text,
            "message_count": message_count,
            "tokens": self.count_tokens(summary_text),
        }

        try:
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)
            logger.debug(f"Stored summary locally for {conversation_id}")
        except Exception as e:
            logger.error(f"Failed to store summary: {e}")

    def _should_update_summary(
        self,
        stored_summary: Optional[Dict],
        current_message_count: int,
    ) -> bool:
        """Check if summary needs to be regenerated."""
        if not stored_summary:
            return True

        stored_count = stored_summary.get("message_count", 0)
        messages_since_summary = current_message_count - stored_count

        return messages_since_summary >= self.config.summary_update_interval

    def build_context(
        self,
        conversation_id: str,
        new_message: str,
        system_prompt: str = None,
        message_count: int = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Build optimized context for API call.

        Returns:
            Tuple of (input_messages, metadata)
            - input_messages: List of message dicts to send to API
            - metadata: Info about context (tokens, summary_used, etc.)

        Structure of returned context:
        1. System message (if provided)
        2. Summary of older messages (if applicable)
        3. Recent window of messages
        4. New user message
        """
        metadata = {
            "mode": self.config.mode,
            "summary_used": False,
            "window_size": 0,
            "total_tokens": 0,
            "summary_tokens": 0,
            "window_tokens": 0,
            "new_message_tokens": 0,
        }

        # If mode is "full", return minimal context (OpenAI handles history)
        if self.config.mode == "full":
            metadata["mode"] = "full"
            metadata["new_message_tokens"] = self.count_tokens(new_message)
            metadata["total_tokens"] = metadata["new_message_tokens"]
            # In full mode, just return the new message
            # The caller will use conversation_id parameter
            return [{"role": "user", "content": new_message}], metadata

        # Summary + Window mode
        messages = []
        total_tokens = 0

        # 1. Add system prompt if provided
        system_tokens = 0
        if system_prompt:
            system_tokens = self.count_tokens(system_prompt)
            total_tokens += system_tokens

        # 2. Fetch history from OpenAI
        history = self._fetch_history_from_openai(conversation_id)
        actual_message_count = len(history)
        logger.debug(f"Fetched {actual_message_count} messages from OpenAI for {conversation_id}")

        if message_count is None:
            message_count = actual_message_count

        # 3. Check if we need summary (enough messages)
        summary_text = ""
        logger.debug(f"Summary check: actual_count={actual_message_count}, threshold={self.config.summarize_after_messages}")
        if actual_message_count >= self.config.summarize_after_messages:
            # Get stored summary
            stored_summary = self._get_stored_summary(conversation_id)
            logger.debug(f"Stored summary: {stored_summary is not None}")

            # Check if we need to update summary
            should_update = self._should_update_summary(stored_summary, actual_message_count)
            logger.debug(f"Should update summary: {should_update}")
            if should_update:
                # Get messages to summarize
                to_summarize = self._get_messages_to_summarize(
                    history,
                    self.config.window_size
                )
                logger.debug(f"Messages to summarize: {len(to_summarize)}")

                if to_summarize:
                    logger.info(f"Generating summary from {len(to_summarize)} messages...")
                    summary_text = self._generate_summary(to_summarize)
                    if summary_text:
                        self._store_summary(conversation_id, summary_text, actual_message_count)
                        logger.info(f"Summary generated and stored: {len(summary_text)} chars")
            else:
                summary_text = stored_summary.get("summary", "") if stored_summary else ""

            if summary_text:
                metadata["summary_used"] = True
                summary_tokens = self.count_tokens(summary_text)
                metadata["summary_tokens"] = summary_tokens
                total_tokens += summary_tokens

                # Add summary as a system-like context message
                messages.append({
                    "role": "user",
                    "content": f"[Previous Conversation Summary]\n{summary_text}\n[End Summary]"
                })

        # 4. Get window of recent messages
        window_messages = self._get_window_messages(history, self.config.window_size)
        window_tokens = self.count_messages_tokens(window_messages)
        metadata["window_size"] = len(window_messages)
        metadata["window_tokens"] = window_tokens
        total_tokens += window_tokens

        # Add window messages
        messages.extend(window_messages)

        # 5. Add new message
        new_message_tokens = self.count_tokens(new_message)
        metadata["new_message_tokens"] = new_message_tokens
        total_tokens += new_message_tokens

        messages.append({"role": "user", "content": new_message})

        # 6. Update total tokens
        metadata["total_tokens"] = total_tokens + system_tokens

        # Log context info
        logger.info(
            f"Built context: {metadata['total_tokens']} tokens "
            f"(summary={metadata['summary_tokens']}, window={metadata['window_tokens']}, "
            f"new={metadata['new_message_tokens']})"
        )

        return messages, metadata

    def build_context_for_api(
        self,
        conversation_id: str,
        new_message: str,
        system_prompt: str = None,
        message_count: int = None,
    ) -> Dict[str, Any]:
        """
        Build context and return parameters for API call.

        Returns dict with:
        - input: The input to pass to create_response
        - use_conversation_id: Whether to use conversation_id parameter
        - metadata: Context metadata
        """
        if self.config.mode == "full":
            # Full mode: let OpenAI handle context
            return {
                "input": new_message,
                "use_conversation_id": True,
                "metadata": {
                    "mode": "full",
                    "total_tokens": self.count_tokens(new_message),
                }
            }

        # Summary + Window mode
        messages, metadata = self.build_context(
            conversation_id,
            new_message,
            system_prompt,
            message_count
        )

        # Convert messages to input format
        # For Responses API, we need to format appropriately
        if len(messages) == 1:
            # Single message - pass as string
            input_data = messages[0]["content"]
        else:
            # Multiple messages - pass as list
            input_data = messages

        return {
            "input": input_data,
            "use_conversation_id": False,  # Don't use conversation param
            "metadata": metadata
        }
