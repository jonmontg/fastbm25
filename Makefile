.PHONY: help install dev-install build test clean format lint

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in production mode
	poetry install --only=main

dev-install: ## Install the package in development mode with all dependencies
	poetry install

build: ## Build the package
	poetry run maturin build --release

build-dev: ## Build the package in development mode
	poetry run maturin develop

test: ## Run tests
	poetry run python test_module.py

format: ## Format code with black and isort
	poetry run black .
	poetry run isort .

lint: ## Run linting with mypy
	poetry run mypy .

clean: ## Clean build artifacts
	rm -rf target/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

publish: ## Publish to PyPI
	poetry run maturin publish

check: format lint test ## Run all checks (format, lint, test)
