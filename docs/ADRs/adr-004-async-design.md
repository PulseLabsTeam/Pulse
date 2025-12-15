# ADR-004: Async-First Design

**Status**: Accepted  
**Date**: 2024-02-01  
**Deciders**: PulseOS Team  
**Tags**: architecture, async, concurrency

## Context

PulseOS needs to handle multiple agents concurrently. Some operations (I/O, network, external services) are blocking. We need a design that supports both synchronous and asynchronous operations efficiently.

## Decision

Design PulseOS with async/await as the primary concurrency model. All runtime operations are async, with sync wrappers where needed.

## Rationale

1. **Scalability**: Async allows handling many agents without threads
2. **Performance**: No GIL contention, efficient I/O
3. **Modern Python**: Async/await is standard in modern Python
4. **Future-Proof**: Supports distributed runtime (future feature)
5. **I/O Operations**: Snapshots, metrics export can be async

## Alternatives Considered

1. **Synchronous Only**
   - Pros: Simpler, no async complexity
   - Cons: Limited scalability, blocking operations

2. **Threading**
   - Pros: Familiar, works with blocking code
   - Cons: GIL limitations, thread overhead, harder to debug

3. **Multiprocessing**
   - Pros: True parallelism, no GIL
   - Cons: High overhead, complex state sharing

4. **Async-First with Sync Wrappers**
   - Pros: Best of both worlds, scalable, modern
   - Cons: Requires async knowledge
   - **Chosen**: Best for our use case

## Consequences

### Positive
- Scalable to thousands of agents
- Efficient I/O operations
- Supports future distributed runtime
- Modern Python best practices

### Negative
- Requires async/await knowledge
- Some complexity in async code
- Need to handle async context properly

### Neutral
- Can wrap async code in sync wrappers if needed

## Implementation Notes

- All runtime operations are async
- Agent.step() is async
- Snapshot operations are async
- Sync wrappers provided for simple use cases
- Proper async context management

## References

- Python asyncio documentation
- TECHNICAL.md - Async Design section
- examples/ - All examples use async

