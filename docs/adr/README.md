# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) documenting key design decisions in PulseOS.

## What are ADRs?

Architecture Decision Records are documents that capture important architectural decisions along with their context and consequences. They help maintain institutional memory and provide rationale for design choices.

## ADR Format

Each ADR follows this structure:

1. **Title**: Short descriptive title
2. **Status**: Proposed, Accepted, Deprecated, Superseded
3. **Context**: The issue motivating this decision
4. **Decision**: The change we're proposing or have made
5. **Consequences**: What becomes easier or harder

## ADR Index

### Core Architecture

- [ADR-001: Survival-Pressure Learning Paradigm](adr-001-survival-pressure-paradigm.md)
- [ADR-002: Circuit-Based Architecture](adr-002-circuit-architecture.md)
- [ADR-003: Async-First Design](adr-003-async-design.md)

### Performance

- [ADR-004: Gradient Caching Strategy](adr-004-gradient-caching.md)
- [ADR-005: Hardware Optimization Layer](adr-005-hardware-optimization.md)
- [ADR-006: Memory Pool Allocation](adr-006-memory-pools.md)

### Reliability

- [ADR-007: Circuit Breaker Pattern](adr-007-circuit-breaker.md)
- [ADR-008: State Persistence and Rollback](adr-008-state-persistence.md)
- [ADR-009: Delta Encoding for Snapshots](adr-009-delta-encoding.md)

### Telemetry

- [ADR-010: Enhanced Metrics Collection](adr-010-enhanced-metrics.md)
- [ADR-011: Prometheus Integration](adr-011-prometheus.md)

## Creating a New ADR

1. Copy `template.md` to `adr-XXX-title.md`
2. Fill in the details
3. Update this index
4. Submit as part of your PR

## References

- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR GitHub Template](https://github.com/joelparkerhenderson/architecture-decision-record)

