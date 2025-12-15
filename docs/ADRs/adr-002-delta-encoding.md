# ADR-002: Delta Encoding for Snapshots

**Status**: Accepted  
**Date**: 2024-01-20  
**Deciders**: PulseOS Team  
**Tags**: persistence, optimization, storage

## Context

State snapshots are critical for rollback and recovery, but storing full snapshots for every checkpoint is storage-intensive. For large-scale deployments with thousands of agents, this becomes prohibitive.

## Decision

Implement delta encoding for snapshots, storing only changes relative to the previous snapshot.

## Rationale

1. **Storage Efficiency**: Patent specifies 70-85% storage reduction target
2. **Scalability**: Enables storing more snapshots with same storage budget
3. **Performance**: Faster snapshot creation (less data to serialize)
4. **Patent Compliance**: Patent explicitly mentions delta encoding

## Alternatives Considered

1. **Full Snapshots Only**
   - Pros: Simpler implementation, faster restore
   - Cons: High storage usage, doesn't meet patent target

2. **Compression Only**
   - Pros: Simpler than delta encoding
   - Cons: Less effective (typically 20-30% vs 70-85%)

3. **Incremental Snapshots (Periodic Full)**
   - Pros: Balance between storage and restore speed
   - Cons: More complex, still higher storage than delta

4. **Delta + Compression**
   - Pros: Maximum storage reduction (80-90%)
   - Cons: More complex, slower restore
   - **Chosen**: Best balance for our use case

## Consequences

### Positive
- 70-85% storage reduction (validated)
- Can store more snapshots
- Faster snapshot creation
- Meets patent specification

### Negative
- Slower restore (must decode delta chain)
- More complex implementation
- Requires maintaining snapshot chain

### Neutral
- Restore time is acceptable for our use case (< 100ms for typical snapshots)

## Implementation Notes

- Delta encoding is recursive for nested dictionaries
- Compression (gzip) applied after delta encoding
- Snapshot chain maintained in circular buffer
- Can disable delta encoding for debugging

## References

- Patent specification: State Persistence and Rollback Subsystem
- TECHNICAL.md - Delta Encoding section
- tests/test_performance.py - Storage reduction validation

