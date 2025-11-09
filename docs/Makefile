# Makefile for building PulseOS documentation

.PHONY: help docs html clean

help:
	@echo "PulseOS Documentation Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  make docs      - Build HTML documentation"
	@echo "  make html      - Alias for docs"
	@echo "  make clean     - Clean build artifacts"

docs html:
	@echo "Building documentation..."
	cd docs && sphinx-build -b html . _build/html
	@echo "Documentation built in docs/_build/html/"

clean:
	@echo "Cleaning build artifacts..."
	rm -rf docs/_build
	rm -rf docs/.doctrees
	@echo "Clean complete"

