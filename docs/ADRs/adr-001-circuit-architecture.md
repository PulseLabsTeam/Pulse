# ADR-001: Circuit-Based Architecture

**Status**: Accepted  
**Date**: 2024-01-15  
**Deciders**: PulseOS Team  
**Tags**: architecture, design-pattern, circuits

## Context

PulseOS implements patent-specified survival-pressure learning algorithms. The patent describes the system in terms of hardware circuits (PTDC, NGCM, APC), but we need to implement this in software.

## Decision

Implement PulseOS using a circuit-based architecture where each patent-specified circuit is a separate Python module with a clear interface.

## Rationale

1. **Patent Alignment**: The patent describes circuits, so our architecture should mirror this structure
2. **Separation of Concerns**: Each circuit has a single responsibility (threshold detection, gradient computation, parameter control)
3. **Testability**: Circuits can be tested independently
4. **Extensibility**: New circuits can be added without modifying existing ones
5. **Clarity**: Code structure matches documentation and patent specifications

## Alternatives Considered

1. **Monolithic Implementation**
   - Pros: Simpler structure, fewer files
   - Cons: Harder to test, doesn't match patent structure, harder to extend

2. **Object-Oriented Classes**
   - Pros: More Pythonic, better encapsulation
   - Cons: Less aligned with patent terminology, harder to map to patent sections

3. **Functional Approach**
   - Pros: Pure functions, easier to test
   - Cons: Doesn't capture stateful nature of circuits, harder to optimize

## Consequences

### Positive
- Code structure matches patent documentation
- Easy to map code to patent sections
- Each circuit can be optimized independently
- Clear separation of concerns
- Easy to test individual circuits

### Negative
- More files to manage
- Need to coordinate between circuits
- Some overhead in circuit communication

### Neutral
- Requires understanding of circuit concept (but this is documented)

## References

- Patent specification sections on PTDC, NGCM, APC
- TECHNICAL.md - Patent Implementation Mapping

