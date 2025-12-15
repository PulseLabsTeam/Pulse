# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) documenting key design decisions in PulseOS.

## What are ADRs?

ADRs are documents that capture important architectural decisions made during the development of PulseOS. They explain:
- **What** decision was made
- **Why** it was made
- **Alternatives** considered
- **Consequences** of the decision

## ADR Index

1. [ADR-001: Circuit-Based Architecture](./adr-001-circuit-architecture.md)
2. [ADR-002: Delta Encoding for Snapshots](./adr-002-delta-encoding.md)
3. [ADR-003: Gradient Caching Strategy](./adr-003-gradient-caching.md)
4. [ADR-004: Async-First Design](./adr-004-async-design.md)
5. [ADR-005: Hardware-Inspired Optimizations](./adr-005-hardware-optimizations.md)

---

## ADR Template

When creating a new ADR, use this template:

```markdown
# ADR-XXX: [Title]

**Status**: [Proposed | Accepted | Deprecated | Superseded]  
**Date**: YYYY-MM-DD  
**Deciders**: [Names]  
**Tags**: [tag1, tag2]

## Context

[Describe the issue motivating this decision]

## Decision

[State the decision]

## Rationale

[Explain why this decision was made]

## Alternatives Considered

1. [Alternative 1]
   - Pros: ...
   - Cons: ...
2. [Alternative 2]
   - Pros: ...
   - Cons: ...

## Consequences

### Positive
- ...

### Negative
- ...

### Neutral
- ...

## References

- [Link to related documents]
```

---

## Contributing ADRs

When making significant architectural decisions:

1. Create a new ADR using the template above
2. Number it sequentially (ADR-006, ADR-007, etc.)
3. Update this index
4. Submit as part of your PR

## ADR Status

- **Proposed**: Decision is under consideration
- **Accepted**: Decision has been made and implemented
- **Deprecated**: Decision has been superseded or is no longer valid
- **Superseded**: Decision has been replaced by a newer ADR

