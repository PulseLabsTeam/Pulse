"""
Performance Profiler

Profiles runtime performance and identifies bottlenecks.
"""

import time
import cProfile
import pstats
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque


@dataclass
class ProfileEntry:
    """Performance profile entry"""
    function_name: str
    call_count: int
    total_time: float
    cumulative_time: float
    per_call_time: float


class PerformanceProfiler:
    """
    Performance profiler for runtime analysis.
    
    Features:
    - Function-level profiling
    - Bottleneck identification
    - Performance regression detection
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self.profiler = cProfile.Profile() if enabled else None
        
        # Manual timing for specific operations
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.active_timers: Dict[str, float] = {}
    
    def start(self) -> None:
        """Start profiling."""
        if self.enabled and self.profiler:
            self.profiler.enable()
    
    def stop(self) -> None:
        """Stop profiling."""
        if self.enabled and self.profiler:
            self.profiler.disable()
    
    def start_timer(self, operation: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation: Operation identifier
        """
        self.active_timers[operation] = time.perf_counter()
    
    def stop_timer(self, operation: str) -> float:
        """
        Stop timing an operation and return duration.
        
        Args:
            operation: Operation identifier
            
        Returns:
            Duration in seconds
        """
        if operation not in self.active_timers:
            return 0.0
        
        duration = time.perf_counter() - self.active_timers[operation]
        self.timings[operation].append(duration)
        del self.active_timers[operation]
        
        return duration
    
    def get_profile_stats(self, top_n: int = 20) -> List[ProfileEntry]:
        """
        Get top N profiled functions.
        
        Args:
            top_n: Number of top functions to return
            
        Returns:
            List of profile entries
        """
        if not self.enabled or not self.profiler:
            return []
        
        stats = pstats.Stats(self.profiler)
        entries = []
        
        for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
            entries.append(ProfileEntry(
                function_name=f"{func_name[0]}:{func_name[1]}({func_name[2]})",
                call_count=cc,
                total_time=tt,
                cumulative_time=ct,
                per_call_time=tt / cc if cc > 0 else 0
            ))
        
        # Sort by total time
        entries.sort(key=lambda x: x.total_time, reverse=True)
        
        return entries[:top_n]
    
    def get_timing_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for timed operations."""
        stats = {}
        
        for operation, timings in self.timings.items():
            if timings:
                stats[operation] = {
                    "mean": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "total": sum(timings),
                    "count": len(timings)
                }
        
        return stats
    
    def get_bottlenecks(self, threshold_percent: float = 10.0) -> List[str]:
        """
        Identify performance bottlenecks.
        
        Args:
            threshold_percent: Threshold percentage of total time
            
        Returns:
            List of bottleneck function names
        """
        if not self.enabled or not self.profiler:
            return []
        
        stats = pstats.Stats(self.profiler)
        total_time = stats.total_tt
        
        if total_time == 0:
            return []
        
        bottlenecks = []
        
        for func_name, (cc, nc, tt, ct, callers) in stats.stats.items():
            percent = (tt / total_time) * 100
            if percent >= threshold_percent:
                bottlenecks.append(f"{func_name[0]}:{func_name[1]}({func_name[2]})")
        
        return bottlenecks
    
    def reset(self) -> None:
        """Reset profiler state."""
        if self.profiler:
            self.profiler = cProfile.Profile()
        self.timings.clear()
        self.active_timers.clear()

