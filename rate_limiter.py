"""
Rate Limiter for Terminal Chatbot
Implements token bucket algorithm for request rate limiting.
"""

import time
from collections import deque
from threading import Lock
from typing import Optional

from exceptions import RateLimitExceededError


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter.

    Limits the number of requests per time window using the token bucket algorithm.
    """

    def __init__(
        self,
        requests_per_minute: int = 10,
        burst_allowance: int = 2
    ):
        """
        Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
            burst_allowance: Extra requests allowed for short bursts
        """
        self.requests_per_minute = requests_per_minute
        self.burst_allowance = burst_allowance
        self.window_size = 60.0  # 1 minute in seconds

        # Token bucket state
        self._request_times: deque = deque()
        self._lock = Lock()

    def _cleanup_old_requests(self, current_time: float) -> None:
        """Remove requests older than the window size."""
        cutoff = current_time - self.window_size
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()

    def _get_current_count(self) -> int:
        """Get the number of requests in the current window."""
        return len(self._request_times)

    def can_proceed(self) -> bool:
        """
        Check if a request can proceed without consuming a token.

        Returns:
            True if request would be allowed, False otherwise
        """
        with self._lock:
            current_time = time.time()
            self._cleanup_old_requests(current_time)
            max_allowed = self.requests_per_minute + self.burst_allowance
            return self._get_current_count() < max_allowed

    def acquire(self, block: bool = False, timeout: float = None) -> bool:
        """
        Attempt to acquire a token for making a request.

        Args:
            block: If True, wait until a token is available
            timeout: Maximum time to wait (only if block=True)

        Returns:
            True if token acquired, False if rate limited

        Raises:
            RateLimitExceededError: If block=False and rate limit exceeded
        """
        start_time = time.time()
        max_allowed = self.requests_per_minute + self.burst_allowance

        while True:
            with self._lock:
                current_time = time.time()
                self._cleanup_old_requests(current_time)

                if self._get_current_count() < max_allowed:
                    self._request_times.append(current_time)
                    return True

                if not block:
                    # Calculate retry_after based on oldest request in window
                    if self._request_times:
                        oldest = self._request_times[0]
                        retry_after = (oldest + self.window_size) - current_time
                    else:
                        retry_after = 1.0

                    raise RateLimitExceededError(
                        message="Rate limit exceeded. Please wait before sending more messages.",
                        requests_per_minute=self.requests_per_minute,
                        retry_after=max(0, retry_after)
                    )

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise RateLimitExceededError(
                        message="Rate limit timeout exceeded",
                        requests_per_minute=self.requests_per_minute,
                        retry_after=self.window_size / max_allowed
                    )

            # Wait a bit before retrying
            time.sleep(0.1)

    def get_remaining(self) -> int:
        """
        Get the number of requests remaining in the current window.

        Returns:
            Number of requests that can still be made
        """
        with self._lock:
            self._cleanup_old_requests(time.time())
            max_allowed = self.requests_per_minute + self.burst_allowance
            return max(0, max_allowed - self._get_current_count())

    def get_reset_time(self) -> Optional[float]:
        """
        Get the time until the rate limit resets (oldest request expires).

        Returns:
            Seconds until reset, or None if no requests in window
        """
        with self._lock:
            current_time = time.time()
            self._cleanup_old_requests(current_time)

            if not self._request_times:
                return None

            oldest = self._request_times[0]
            reset_time = (oldest + self.window_size) - current_time
            return max(0, reset_time)

    def reset(self) -> None:
        """Reset the rate limiter state (clear all request history)."""
        with self._lock:
            self._request_times.clear()

    @property
    def is_limited(self) -> bool:
        """Check if currently rate limited."""
        return not self.can_proceed()

    def __repr__(self) -> str:
        remaining = self.get_remaining()
        return (
            f"TokenBucketRateLimiter("
            f"requests_per_minute={self.requests_per_minute}, "
            f"remaining={remaining})"
        )
