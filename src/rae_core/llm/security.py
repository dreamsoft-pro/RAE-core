import os
import re
import time
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

KEY_PATTERN = re.compile(r"(sk-[a-zA-Z0-9]{32,}|x-api-key:[a-zA-Z0-9_-]{20,})")

def redact_secrets(text: str) -> str:
    if not text:
        return text
    text = KEY_PATTERN.sub("[REDACTED_API_KEY]", text)
    
    for key_env in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY", "QWEN_API_KEY"]:
        val = os.getenv(key_env)
        if val and len(val) > 8:
            text = text.replace(val, "[REDACTED]")
            
    return text

class CircuitBreakerOpenException(Exception):
    """Exception raised when the circuit breaker is open."""
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_state_change = time.time()

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_state_change = time.time()
            logger.warning(f"CircuitBreaker tripped to OPEN. Threshold: {self.failure_threshold} failures.")

    def check_request_allowed(self):
        if self.state == "OPEN":
            if time.time() - self.last_state_change > self.recovery_timeout:
                self.state = "HALF-OPEN"
                logger.info("CircuitBreaker transitioned to HALF-OPEN. Attempting trial request.")
            else:
                raise CircuitBreakerOpenException("Circuit breaker is OPEN. Request blocked.")

class StandaloneRateLimiter:
    def __init__(self, max_requests: int = 15, period: float = 60.0):
        self.max_requests = max_requests
        self.period = period
        self.requests: list[float] = []

    def allow_request(self) -> bool:
        now = time.time()
        self.requests = [r for r in self.requests if now - r < self.period]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
