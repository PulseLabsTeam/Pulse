"""
Core Runtime Orchestrator

Coordinates all subsystems and implements the main survival-pressure learning loop.
Implements clean architecture with dependency injection and event-driven design.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

from pulseos.circuits.ptdc import PerformanceThresholdDetectionCircuit
from pulseos.circuits.ngcm import NonlinearGradientComputationModule
from pulseos.circuits.apc import AdaptiveParameterController
from pulseos.persistence.snapshot import SnapshotManager
from pulseos.agent import Agent, SurvivalConstraint
from pulseos.telemetry.metrics import MetricsCollector
from pulseos.telemetry.profiler import PerformanceProfiler


class RuntimeState(Enum):
    """Runtime state machine"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ROLLING_BACK = "rolling_back"
    SHUTTING_DOWN = "shutting_down"
    ERROR = "error"


@dataclass
class Config:
    """Runtime configuration with all tunable parameters"""
    
    # PTDC Configuration
    threshold_detection_interval: float = 0.001  # Sub-millisecond detection
    normalization_window: int = 100
    
    # NGCM Configuration
    gradient_cache_size: int = 256
    cache_implementation: str = "LUT"  # LUT, PLA, or CORDIC
    beta_parameter: float = 1.0
    target_cache_hit_rate: float = 0.75
    
    # APC Configuration
    alpha_base: float = 0.01
    alpha_max_change_per_step: float = 0.10  # 10% max change
    alpha_smooth: float = 0.9  # EMA smoothing
    epsilon_min: float = 0.01
    epsilon_max: float = 0.3
    epsilon_kappa: float = 2.0
    gamma: float = 0.1
    
    # SPRS Configuration
    snapshot_interval: float = 1.0
    max_snapshots: int = 100
    rollback_grace_period: float = 5.0
    critical_survival_threshold: float = 0.3
    enable_delta_encoding: bool = True
    enable_compression: bool = True
    
    # Performance Configuration
    max_agents: int = 10000
    vectorization_enabled: bool = True
    parallel_updates: bool = True
    update_batch_size: int = 100
    
    # Telemetry Configuration
    metrics_enabled: bool = True
    profiling_enabled: bool = True
    metrics_export_interval: float = 10.0


class Runtime:
    """
    Main runtime orchestrator implementing survival-pressure learning.
    
    Coordinates all subsystems and manages the adaptive learning loop.
    Implements event-driven architecture with backpressure handling.
    """
    
    def __init__(
        self,
        constraint: SurvivalConstraint,
        config: Optional[Config] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        profiler: Optional[PerformanceProfiler] = None
    ):
        """
        Initialize the PulseOS runtime.
        
        Args:
            constraint: Survival constraint defining performance thresholds
            config: Runtime configuration (uses defaults if None)
            metrics_collector: Optional metrics collector for telemetry
            profiler: Optional performance profiler
        """
        self.config = config or Config()
        self.constraint = constraint
        self.state = RuntimeState.INITIALIZING
        
        # Initialize core circuits
        self.ptdc = PerformanceThresholdDetectionCircuit(
            threshold=self.constraint.threshold,
            normalization_window=self.config.normalization_window,
            detection_interval=self.config.threshold_detection_interval
        )
        
        self.ngcm = NonlinearGradientComputationModule(
            cache_size=self.config.gradient_cache_size,
            implementation=self.config.cache_implementation,
            beta=self.config.beta_parameter,
            target_hit_rate=self.config.target_cache_hit_rate
        )
        
        self.apc = AdaptiveParameterController(
            alpha_base=self.config.alpha_base,
            alpha_max_change=self.config.alpha_max_change_per_step,
            alpha_smooth=self.config.alpha_smooth,
            epsilon_min=self.config.epsilon_min,
            epsilon_max=self.config.epsilon_max,
            epsilon_kappa=self.config.epsilon_kappa,
            gamma=self.config.gamma
        )
        
        # Initialize persistence subsystem
        self.sprs = SnapshotManager(
            snapshot_interval=self.config.snapshot_interval,
            max_snapshots=self.config.max_snapshots,
            enable_delta_encoding=self.config.enable_delta_encoding,
            enable_compression=self.config.enable_compression
        )
        
        # Telemetry
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.profiler = profiler or PerformanceProfiler()
        
        # Agent management
        self.agents: Dict[str, Agent] = {}
        self.agent_metrics: Dict[str, deque] = {}
        
        # Runtime state
        self.current_step: int = 0
        self.start_time: float = 0.0
        self.last_snapshot_time: float = 0.0
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "threshold_breach": [],
            "rollback": [],
            "convergence": [],
            "error": []
        }
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        
    def register_agent(self, agent_id: str, agent: Agent) -> None:
        """
        Register an agent with the runtime.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: Agent instance implementing the Agent interface
        """
        if len(self.agents) >= self.config.max_agents:
            raise RuntimeError(f"Maximum agent limit ({self.config.max_agents}) reached")
        
        self.agents[agent_id] = agent
        self.agent_metrics[agent_id] = deque(maxlen=self.config.normalization_window)
        
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the runtime."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            del self.agent_metrics[agent_id]
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler for runtime events."""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit an event to all registered handlers."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                # Log error but don't crash runtime
                print(f"Error in event handler for {event_type}: {e}")
    
    async def step(self) -> Dict[str, Any]:
        """
        Execute one step of the survival-pressure learning loop.
        
        Returns:
            Dictionary containing step results and metrics
        """
        if self.state != RuntimeState.RUNNING:
            raise RuntimeError(f"Cannot execute step in state: {self.state}")
        
        step_start_time = time.perf_counter()
        self.current_step += 1
        
        # Collect current performance metrics from all agents
        current_metrics = self._collect_agent_metrics()
        
        # Update PTDC with current metrics
        threshold_status = self.ptdc.evaluate(current_metrics)
        
        # Compute survival pressure signal
        survival_signal = self._compute_survival_signal(threshold_status)
        
        # Compute gradient via NGCM
        gradient = self.ngcm.compute_gradient(
            delta=survival_signal,
            timestamp=self.current_step
        )
        
        # Update adaptive parameters via APC
        alpha, epsilon = self.apc.update_parameters(gradient, survival_signal)
        
        # Update all agents with new parameters
        update_results = await self._update_agents(alpha, epsilon)
        
        # Check if snapshot is needed
        if self._should_create_snapshot():
            await self._create_snapshot()
        
        # Check if rollback is needed
        if self._should_rollback(survival_signal):
            await self._execute_rollback()
        
        # Update performance history
        step_duration = time.perf_counter() - step_start_time
        self.performance_history.append({
            "step": self.current_step,
            "duration": step_duration,
            "survival_signal": survival_signal,
            "alpha": alpha,
            "epsilon": epsilon,
            "gradient": gradient,
            "threshold_status": threshold_status
        })
        
        # Collect metrics
        if self.config.metrics_enabled:
            self.metrics_collector.record_step(
                step=self.current_step,
                duration=step_duration,
                survival_signal=survival_signal,
                alpha=alpha,
                epsilon=epsilon,
                gradient=gradient,
                agent_count=len(self.agents)
            )
        
        return {
            "step": self.current_step,
            "survival_signal": survival_signal,
            "alpha": alpha,
            "epsilon": epsilon,
            "gradient": gradient,
            "threshold_status": threshold_status,
            "update_results": update_results,
            "duration": step_duration
        }
    
    def _collect_agent_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics from all agents."""
        metrics = {}
        
        for agent_id, agent in self.agents.items():
            try:
                agent_metric = agent.get_performance_metric()
                metrics[agent_id] = agent_metric
                self.agent_metrics[agent_id].append(agent_metric)
            except Exception as e:
                # Log error but continue
                print(f"Error collecting metrics from agent {agent_id}: {e}")
                metrics[agent_id] = 0.0
        
        return metrics
    
    def _compute_survival_signal(self, threshold_status: Dict[str, bool]) -> float:
        """
        Compute survival pressure signal from threshold status.
        
        Returns:
            Survival signal value between 0 and 1
        """
        if not threshold_status:
            return 0.0
        
        # Compute fraction of agents meeting threshold
        meeting_threshold = sum(1 for v in threshold_status.values() if v)
        total_agents = len(threshold_status)
        
        if total_agents == 0:
            return 0.0
        
        survival_ratio = meeting_threshold / total_agents
        
        # Apply constraint-specific computation
        return self.constraint.compute_survival_signal(survival_ratio)
    
    async def _update_agents(self, alpha: float, epsilon: float) -> Dict[str, Any]:
        """Update all agents with new adaptive parameters."""
        results = {}
        
        if self.config.parallel_updates and len(self.agents) > self.config.update_batch_size:
            # Batch parallel updates
            agent_list = list(self.agents.items())
            batches = [
                agent_list[i:i + self.config.update_batch_size]
                for i in range(0, len(agent_list), self.config.update_batch_size)
            ]
            
            for batch in batches:
                tasks = [
                    self._update_single_agent(agent_id, agent, alpha, epsilon)
                    for agent_id, agent in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for (agent_id, _), result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        results[agent_id] = {"error": str(result)}
                    else:
                        results[agent_id] = result
        else:
            # Sequential updates
            for agent_id, agent in self.agents.items():
                try:
                    results[agent_id] = await self._update_single_agent(
                        agent_id, agent, alpha, epsilon
                    )
                except Exception as e:
                    results[agent_id] = {"error": str(e)}
        
        return results
    
    async def _update_single_agent(
        self,
        agent_id: str,
        agent: Agent,
        alpha: float,
        epsilon: float
    ) -> Dict[str, Any]:
        """Update a single agent with new parameters."""
        update_start = time.perf_counter()
        
        # Update agent parameters
        agent.update_learning_rate(alpha)
        agent.update_exploration_rate(epsilon)
        
        # Execute agent step
        agent_result = await agent.step()
        
        update_duration = time.perf_counter() - update_start
        
        return {
            "agent_id": agent_id,
            "result": agent_result,
            "duration": update_duration
        }
    
    def _should_create_snapshot(self) -> bool:
        """Determine if a snapshot should be created."""
        current_time = time.time()
        time_since_snapshot = current_time - self.last_snapshot_time
        
        return time_since_snapshot >= self.config.snapshot_interval
    
    async def _create_snapshot(self) -> None:
        """Create a state snapshot."""
        snapshot_data = {
            "step": self.current_step,
            "timestamp": time.time(),
            "agents": {
                agent_id: agent.get_state()
                for agent_id, agent in self.agents.items()
            },
            "apc_state": self.apc.get_state(),
            "ngcm_state": self.ngcm.get_state(),
            "ptdc_state": self.ptdc.get_state()
        }
        
        await self.sprs.create_snapshot(snapshot_data)
        self.last_snapshot_time = time.time()
    
    def _should_rollback(self, survival_signal: float) -> bool:
        """Determine if rollback is needed based on survival signal."""
        if survival_signal >= self.config.critical_survival_threshold:
            return False
        
        # Check if signal has been below threshold for grace period
        recent_signals = [
            h["survival_signal"]
            for h in list(self.performance_history)[-int(self.config.rollback_grace_period * 10):]
        ]
        
        if len(recent_signals) < 5:
            return False
        
        all_below_threshold = all(
            s < self.config.critical_survival_threshold
            for s in recent_signals
        )
        
        return all_below_threshold
    
    async def _execute_rollback(self) -> None:
        """Execute state rollback to best recovery snapshot."""
        self.state = RuntimeState.ROLLING_BACK
        
        try:
            # Find best recovery snapshot
            best_snapshot = await self.sprs.find_best_recovery_snapshot()
            
            if best_snapshot is None:
                print("No recovery snapshot available")
                self.state = RuntimeState.ERROR
                self._emit_event("error", {"type": "rollback_failed", "reason": "no_snapshot"})
                return
            
            # Restore state
            await self._restore_snapshot(best_snapshot)
            
            # Increase exploration for recovery
            self.apc.increase_exploration()
            
            self.state = RuntimeState.RUNNING
            self._emit_event("rollback", {"snapshot": best_snapshot, "step": self.current_step})
            
        except Exception as e:
            self.state = RuntimeState.ERROR
            self._emit_event("error", {"type": "rollback_failed", "error": str(e)})
    
    async def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore runtime state from snapshot."""
        self.current_step = snapshot["step"]
        
        # Restore agent states
        for agent_id, agent_state in snapshot.get("agents", {}).items():
            if agent_id in self.agents:
                self.agents[agent_id].restore_state(agent_state)
        
        # Restore subsystem states
        if "apc_state" in snapshot:
            self.apc.restore_state(snapshot["apc_state"])
        if "ngcm_state" in snapshot:
            self.ngcm.restore_state(snapshot["ngcm_state"])
        if "ptdc_state" in snapshot:
            self.ptdc.restore_state(snapshot["ptdc_state"])
    
    async def run(self, max_steps: Optional[int] = None) -> None:
        """
        Run the survival-pressure learning loop.
        
        Args:
            max_steps: Maximum number of steps to run (None for infinite)
        """
        self.state = RuntimeState.RUNNING
        self.start_time = time.time()
        self.last_snapshot_time = time.time()
        
        try:
            step_count = 0
            while self.state == RuntimeState.RUNNING:
                if max_steps is not None and step_count >= max_steps:
                    break
                
                await self.step()
                step_count += 1
                
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.001)
                
        except KeyboardInterrupt:
            print("Runtime interrupted by user")
        except Exception as e:
            self.state = RuntimeState.ERROR
            self._emit_event("error", {"type": "runtime_error", "error": str(e)})
            raise
        finally:
            self.state = RuntimeState.SHUTTING_DOWN
    
    def pause(self) -> None:
        """Pause the runtime."""
        self.state = RuntimeState.PAUSED
    
    def resume(self) -> None:
        """Resume the runtime."""
        if self.state == RuntimeState.PAUSED:
            self.state = RuntimeState.RUNNING
    
    def shutdown(self) -> None:
        """Shutdown the runtime gracefully."""
        self.state = RuntimeState.SHUTTING_DOWN
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get runtime statistics and performance metrics."""
        if not self.performance_history:
            return {}
        
        recent_history = list(self.performance_history)
        
        return {
            "current_step": self.current_step,
            "agent_count": len(self.agents),
            "uptime": time.time() - self.start_time if self.start_time > 0 else 0,
            "average_step_duration": np.mean([h["duration"] for h in recent_history]),
            "average_survival_signal": np.mean([h["survival_signal"] for h in recent_history]),
            "current_alpha": recent_history[-1]["alpha"] if recent_history else 0,
            "current_epsilon": recent_history[-1]["epsilon"] if recent_history else 0,
            "ngcm_cache_hit_rate": self.ngcm.get_cache_hit_rate(),
            "snapshot_count": self.sprs.get_snapshot_count(),
            "rollback_count": len([h for h in recent_history if "rollback" in str(h)])
        }

