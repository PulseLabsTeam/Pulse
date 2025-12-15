# PulseOS API Documentation

This directory contains comprehensive API documentation for PulseOS.

## Structure

- **API Reference**: Complete API documentation (Sphinx-generated)
- **Tutorials**: Step-by-step guides for common tasks
- **Examples**: Code examples for various use cases

## Quick Links

- [Core API](api/core.md) - Runtime, Agent, Config
- [Circuits API](api/circuits.md) - PTDC, NGCM, APC
- [Persistence API](api/persistence.md) - Snapshots, Rollback
- [Telemetry API](api/telemetry.md) - Metrics, Profiling
- [Optimization API](api/optimization.md) - Caching, Hardware

## Generating API Documentation

To generate Sphinx documentation:

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Generate documentation
cd docs
sphinx-build -b html . _build/html
```

## Documentation Standards

- All public APIs must be documented
- Include type hints for all functions
- Provide examples for complex APIs
- Document performance characteristics
- Note any limitations or gotchas

## Contributing

When adding new APIs:

1. Add docstrings following Google style
2. Include type hints
3. Add examples to this directory
4. Update relevant API reference pages
5. Run Sphinx to verify formatting

