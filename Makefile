.PHONY: install dev test lint format type-check security clean run docs

# Install production dependencies
install:
	pip install -e .

# Install development dependencies
dev:
	pip install -e ".[dev]"
	pre-commit install

# Run all tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Run unit tests only
test-unit:
	pytest tests/unit/ -v -m unit

# Run integration tests only
test-integration:
	pytest tests/integration/ -v -m integration

# Run linting
lint:
	ruff check src/ tests/

# Format code
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Run type checking
type-check:
	mypy src/

# Run security scan
security:
	bandit -r src/ -ll

# Run all checks
check: lint type-check security test

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run the API server
run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run with production settings
run-prod:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Build Docker image
docker-build:
	docker build -t dq-leakage-detection .

# Run Docker container
docker-run:
	docker run -p 8000:8000 dq-leakage-detection

# Generate documentation
docs:
	mkdocs build

# Serve documentation locally
docs-serve:
	mkdocs serve
