"""
Circuit Breaker and Error Handling

Implements production-ready error handling with circuit breakers,
retry logic, and graceful degradation.
"""

from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass, field
from enum import Enum
import time
import asyncio
from collections import deque


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    half_open_timeout: float = 10.0


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    Implements:
    - Failure threshold detection
    - Automatic recovery testing
    - Exponential backoff
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        
        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_state_change = time.time()
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_rejections = 0
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        self.total_requests += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_state_change > self.config.timeout_seconds:
                # Try recovery
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.last_state_change = time.time()
            else:
                self.total_rejections += 1
                raise RuntimeError("Circuit breaker is OPEN")
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Async version of call."""
        self.total_requests += 1
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_state_change > self.config.timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.last_state_change = time.time()
            else:
                self.total_rejections += 1
                raise RuntimeError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.last_state_change = time.time()
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
        elif self.failure_count >= self.config.failure_threshold:
            # Open circuit
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "failure_rate": self.total_failures / self.total_requests if self.total_requests > 0 else 0.0
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = time.time()


class RetryHandler:
    """
    Retry handler with exponential backoff.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(
                        self.initial_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    await asyncio.sleep(delay)
                else:
                    raise last_exception
        
        raise last_exception

