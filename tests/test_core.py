"""
Test suite for core modules

Tests circuit breaker, error handling, and fault tolerance mechanisms.
"""

import pytest
import asyncio
import time
from pulseos.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryHandler
)


class TestCircuitBreaker:
    """Tests for Circuit Breaker"""
    
    def test_initial_state(self):
        """Test circuit breaker starts in CLOSED state"""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
    
    def test_successful_call(self):
        """Test successful function call"""
        cb = CircuitBreaker()
        
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.total_requests == 1
        assert cb.total_failures == 0
    
    def test_failure_tracking(self):
        """Test failure tracking"""
        cb = CircuitBreaker(config=CircuitBreakerConfig(failure_threshold=3))
        
        def failing_func():
            raise ValueError("Test error")
        
        # First 2 failures should not open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2
        
        # Third failure should open circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3
        assert cb.total_failures == 3
    
    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold"""
        cb = CircuitBreaker(config=CircuitBreakerConfig(failure_threshold=2))
        
        def failing_func():
            raise RuntimeError("Error")
        
        # Trigger failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Next call should be rejected
        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            cb.call(lambda: None)
        
        assert cb.total_rejections >= 1
    
    def test_circuit_recovery(self):
        """Test circuit recovery after timeout"""
        cb = CircuitBreaker(config=CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1
        ))
        
        def failing_func():
            raise ValueError("Error")
        
        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Next call should transition to HALF_OPEN
        def success_func():
            return "success"
        
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
    
    def test_half_open_recovery(self):
        """Test recovery from HALF_OPEN to CLOSED"""
        cb = CircuitBreaker(config=CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=0.1
        ))
        
        def failing_func():
            raise ValueError("Error")
        
        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait and recover
        time.sleep(0.15)
        
        def success_func():
            return "success"
        
        # First success -> HALF_OPEN
        cb.call(success_func)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.success_count == 1
        
        # Second success -> CLOSED
        cb.call(success_func)
        assert cb.state == CircuitState.CLOSED
        assert cb.success_count == 0  # Reset after closing
    
    def test_half_open_failure(self):
        """Test failure during HALF_OPEN returns to OPEN"""
        cb = CircuitBreaker(config=CircuitBreakerConfig(
            failure_threshold=2,
            timeout_seconds=0.1
        ))
        
        def failing_func():
            raise ValueError("Error")
        
        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(0.15)
        
        # Failure during HALF_OPEN should return to OPEN
        with pytest.raises(ValueError):
            cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_async_call(self):
        """Test async function call"""
        cb = CircuitBreaker()
        
        async def async_success():
            await asyncio.sleep(0.01)
            return "async_success"
        
        result = await cb.call_async(async_success)
        assert result == "async_success"
        assert cb.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_async_failure(self):
        """Test async function failure"""
        cb = CircuitBreaker(config=CircuitBreakerConfig(failure_threshold=2))
        
        async def async_fail():
            await asyncio.sleep(0.01)
            raise ValueError("Async error")
        
        with pytest.raises(ValueError):
            await cb.call_async(async_fail)
        
        assert cb.failure_count == 1
    
    def test_statistics(self):
        """Test circuit breaker statistics"""
        cb = CircuitBreaker(config=CircuitBreakerConfig(failure_threshold=3))
        
        def success_func():
            return "success"
        
        def failing_func():
            raise ValueError("Error")
        
        # Mix of successes and failures
        cb.call(success_func)
        cb.call(success_func)
        
        with pytest.raises(ValueError):
            cb.call(failing_func)
        
        stats = cb.get_statistics()
        assert stats["total_requests"] == 3
        assert stats["total_failures"] == 1
        assert stats["failure_rate"] == pytest.approx(1.0 / 3.0, abs=0.01)
        assert stats["state"] == CircuitState.CLOSED.value
    
    def test_reset(self):
        """Test circuit breaker reset"""
        cb = CircuitBreaker(config=CircuitBreakerConfig(failure_threshold=2))
        
        def failing_func():
            raise ValueError("Error")
        
        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(failing_func)
        
        assert cb.state == CircuitState.OPEN
        
        # Reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0


class TestRetryHandler:
    """Tests for Retry Handler"""
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful execution without retries"""
        handler = RetryHandler(max_retries=3)
        
        async def success_func():
            return "success"
        
        result = await handler.execute_with_retry(success_func)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on failure"""
        handler = RetryHandler(max_retries=3, initial_delay=0.01)
        attempt_count = [0]
        
        async def failing_func():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = await handler.execute_with_retry(failing_func)
        assert result == "success"
        assert attempt_count[0] == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test failure after max retries"""
        handler = RetryHandler(max_retries=2, initial_delay=0.01)
        
        async def always_failing():
            raise ValueError("Persistent error")
        
        with pytest.raises(ValueError, match="Persistent error"):
            await handler.execute_with_retry(always_failing)
    
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff timing"""
        handler = RetryHandler(
            max_retries=3,
            initial_delay=0.01,
            exponential_base=2.0
        )
        delays = []
        last_time = [time.time()]
        
        async def failing_func():
            current_time = time.time()
            if last_time[0] > 0:
                delays.append(current_time - last_time[0])
            last_time[0] = time.time()
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            await handler.execute_with_retry(failing_func)
        
        # Should have delays increasing exponentially
        assert len(delays) >= 2
        if len(delays) >= 2:
            # Second delay should be ~2x first delay
            assert delays[1] >= delays[0] * 1.5  # Allow some variance
    
    @pytest.mark.asyncio
    async def test_sync_function(self):
        """Test retry handler with sync function"""
        handler = RetryHandler(max_retries=2, initial_delay=0.01)
        attempt_count = [0]
        
        def sync_failing_func():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise ValueError("Error")
            return "success"
        
        result = await handler.execute_with_retry(sync_failing_func)
        assert result == "success"
        assert attempt_count[0] == 2
    
    @pytest.mark.asyncio
    async def test_max_delay_limit(self):
        """Test max delay limit"""
        handler = RetryHandler(
            max_retries=5,
            initial_delay=1.0,
            max_delay=0.1,  # Max delay is less than exponential growth
            exponential_base=2.0
        )
        
        # Should not exceed max_delay even with exponential growth
        assert handler.max_delay == 0.1
        
        # Test that delays are capped
        call_count = [0]
        async def failing_func():
            call_count[0] += 1
            raise ValueError("Error")
        
        start_time = time.time()
        with pytest.raises(ValueError):
            await handler.execute_with_retry(failing_func)
        elapsed = time.time() - start_time
        
        # Should have delays capped at max_delay
        # With max_retries=5, we have 5 retries, so 4 delays
        # Each delay should be <= max_delay (0.1)
        # Total time should be reasonable
        assert elapsed < 1.0  # Should complete quickly with small delays

