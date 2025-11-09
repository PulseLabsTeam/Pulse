# ADR-001: Survival-Pressure Learning Paradigm

**Status**: Accepted  
**Date**: 2024-01-01  
**Deciders**: PulseOS Team

## Context

Traditional reinforcement learning approaches use episodic or continuous reward signals. We needed a framework that:

1. Provides continuous constraint evaluation (not episodic)
2. Adapts learning parameters based on constraint satisfaction
3. Supports multiple constraint types (temporal, statistical, composite)
4. Enables real-time performance requirements

## Decision

We adopted a "survival-pressure" learning paradigm where:

- Agents must maintain performance above thresholds (survival constraints)
- Survival signal (0-1) measures how well agents meet constraints
- Learning rate and exploration rate adapt based on survival signal
- Constraints can be simple, temporal, statistical, or composite

## Consequences

### Positive

- ✅ Continuous constraint evaluation (not episodic)
- ✅ Automatic parameter adaptation
- ✅ Supports multiple constraint types
- ✅ Real-time performance validated (<1ms latency)
- ✅ Novel approach with patent backing

### Negative

- ⚠️ Learning curve for new users (novel concepts)
- ⚠️ Not general-purpose RL (specialized framework)
- ⚠️ Requires understanding of survival-pressure concepts

## Implementation

- `SurvivalConstraint` class in `pulseos/agent.py`
- `Runtime` orchestrates survival-pressure loop
- `AdaptiveParameterController` adapts parameters based on survival signal

## References

- Patent specifications for survival-pressure learning
- TECHNICAL.md for algorithm details

