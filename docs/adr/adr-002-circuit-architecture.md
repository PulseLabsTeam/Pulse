# ADR-002: Circuit-Based Architecture

**Status**: Accepted  
**Date**: 2024-01-01  
**Deciders**: PulseOS Team

## Context

We needed a modular architecture that:

1. Separates concerns cleanly
2. Enables hardware optimization
3. Supports multiple implementation strategies
4. Maintains patent algorithm fidelity

## Decision

We organized core algorithms into "circuits":

- **PTDC** (Performance Threshold Detection Circuit): Threshold evaluation
- **NGCM** (Nonlinear Gradient Computation Module): Gradient computation
- **APC** (Adaptive Parameter Controller): Parameter adaptation

Each circuit:
- Has a clear, single responsibility
- Supports multiple implementation strategies (LUT, PLA, CORDIC, EXACT)
- Can be optimized independently
- Maintains patent-specified interfaces

## Consequences

### Positive

- ✅ Clean separation of concerns
- ✅ Easy to test individual components
- ✅ Supports hardware optimization (FPGA/ASIC ready)
- ✅ Multiple implementation strategies for performance trade-offs
- ✅ Clear module boundaries

### Negative

- ⚠️ More files to maintain
- ⚠️ Need to understand circuit interactions

## Implementation

- `pulseos/circuits/ptdc.py`
- `pulseos/circuits/ngcm.py`
- `pulseos/circuits/apc.py`

## References

- TECHNICAL.md for circuit details
- Patent specifications for algorithm requirements

