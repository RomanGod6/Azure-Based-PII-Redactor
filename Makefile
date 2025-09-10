# PII Redactor Pro - Modern Development Setup
# Cross-platform Makefile for easy project management

.PHONY: help install setup run clean test lint format check dev build

# Default target
help: ## Show this help message
	@echo "PII Redactor Pro - Available Commands:"
	@echo ""
	@echo "🚀 MAIN COMMANDS:"
	@echo "  make run           - Start web app (recommended - opens in browser)"
	@echo "  make run-gui       - Start desktop GUI (requires tkinter)"
	@echo ""
	@echo "🌐 WEB VERSION OPTIONS:"
	@echo "  make web           - Same as 'run' (web app)"
	@echo "  make web-dev       - Web app in development mode"
	@echo ""
	@echo "⚙️  SETUP & DEVELOPMENT:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v "web\|run" | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install Poetry (if not installed) and project dependencies
	@echo "🔧 Setting up PII Redactor Pro..."
	@command -v poetry >/dev/null 2>&1 || { echo "Installing Poetry..."; curl -sSL https://install.python-poetry.org | python3 -; }
	@poetry --version
	@poetry install --no-root
	@echo "✅ Dependencies installed!"

setup: install ## Complete project setup (install + configure environment)
	@echo "📋 Setting up environment..."
	@if [ ! -f .env ]; then \
		if [ -f .env.template ]; then \
			cp .env.template .env; \
			echo "✅ Created .env from template"; \
		else \
			echo "⚠️  No .env.template found"; \
		fi; \
	else \
		echo "✅ .env file already exists"; \
	fi
	@echo ""
	@echo "🎉 Setup complete! Use 'make run' to start the application"

run: ## Run the PII Redactor web application (default)
	@echo "🌐 Starting PII Redactor Pro Web App..."
	@echo "📱 Opening in browser - perfect for WSL!"
	@poetry run streamlit run pii_redactor_web.py --server.port 8501 --server.address 0.0.0.0

run-gui: ## Run the desktop GUI version (requires tkinter)
	@echo "�️ Starting PII Redactor Pro Desktop GUI..."
	@poetry run python -c "import tkinter; print('✅ tkinter available')" 2>/dev/null || { \
		echo "❌ tkinter not found in WSL environment"; \
		echo ""; \
		echo "🔧 To fix this, you have several options:"; \
		echo ""; \
		echo "Option 1 - Install tkinter in WSL:"; \
		echo "  sudo apt update && sudo apt install python3-tk python3-dev"; \
		echo ""; \
		echo "Option 2 - Run from Windows instead:"; \
		echo "  cd /mnt/c/Users/dgriffey/Downloads/pii_redactor_pro"; \
		echo "  python.exe pii_redactor_app.py"; \
		echo ""; \
		echo "Option 3 - Use web version (recommended):"; \
		echo "  make run"; \
		echo ""; \
		echo "Option 4 - Use batch processing (no GUI):"; \
		echo "  make batch ARGS='--dir /path/to/csv/files'"; \
		echo ""; \
		exit 1; \
	}
	@poetry run python pii_redactor_app.py

web: ## Run the modern web-based version (recommended)
	@echo "🌐 Starting PII Redactor Pro Web App..."
	@echo "📱 This will open in your browser - works great in WSL!"
	@poetry run streamlit run pii_redactor_web.py --server.port 8501 --server.address 0.0.0.0

web-dev: ## Run web version in development mode
	@echo "🌐 Starting PII Redactor Pro Web App in dev mode..."
	@poetry run streamlit run pii_redactor_web.py --server.port 8501 --server.address 0.0.0.0 --server.runOnSave true

run-windows: ## Run the app using Windows Python (from WSL)
	@echo "🚀 Starting PII Redactor Pro using Windows Python..."
	@cd /mnt/c/Users/dgriffey/Downloads/pii_redactor_pro && python.exe -c "import sys; print('Python:', sys.version)"
	@cd /mnt/c/Users/dgriffey/Downloads/pii_redactor_pro && python.exe -m pip install -r requirements.txt --quiet || echo "Dependencies may need installing"
	@cd /mnt/c/Users/dgriffey/Downloads/pii_redactor_pro && python.exe pii_redactor_app.py

batch: ## Run batch processing
	@echo "📦 Starting batch processor..."
	@poetry run python batch_process.py $(ARGS)

dev: install ## Install development dependencies and run in dev mode
	@poetry install --with dev --no-root || poetry install --no-root
	@echo "🔧 Development environment ready!"
	@echo "You can now run: make run"

test: ## Run tests
	@echo "🧪 Running tests..."
	@poetry run pytest

lint: ## Run linting checks
	@echo "🔍 Running linting..."
	@poetry run flake8 .
	@poetry run mypy .

format: ## Format code with black and isort
	@echo "✨ Formatting code..."
	@poetry run black .
	@poetry run isort .

check: format lint test ## Run all quality checks (format, lint, test)
	@echo "✅ All checks passed!"

check-env: ## Check if the environment is properly set up
	@echo "🔍 Checking environment..."
	@echo "Python version:"
	@poetry run python --version
	@echo ""
	@echo "Checking tkinter availability:"
	@poetry run python -c "import tkinter; print('✅ tkinter is available')" 2>/dev/null || { \
		echo "❌ tkinter not available in this environment"; \
		echo "   This is common in WSL - see 'make run' for solutions"; \
	}
	@echo ""
	@echo "Checking Azure credentials:"
	@if [ -f .env ]; then \
		echo "✅ .env file exists"; \
		poetry run python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Azure Endpoint:', os.getenv('AZURE_ENDPOINT', 'Not set')); print('Azure Key:', 'Set' if os.getenv('AZURE_KEY') else 'Not set')" 2>/dev/null || echo "Could not load .env"; \
	else \
		echo "❌ .env file not found"; \
	fi

clean: ## Clean up temporary files and caches
	@echo "🧹 Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@rm -rf dist/ build/ .pytest_cache/ .mypy_cache/ .coverage
	@echo "✅ Cleanup complete!"

build: ## Build distribution packages
	@echo "📦 Building package..."
	@poetry build
	@echo "✅ Build complete! Check dist/ directory"

deps-update: ## Update all dependencies to latest versions
	@echo "📦 Updating dependencies..."
	@poetry update
	@echo "✅ Dependencies updated!"

deps-show: ## Show current dependency tree
	@poetry show --tree

shell: ## Open a shell in the virtual environment
	@poetry shell

info: ## Show project and environment info
	@echo "📊 Project Information:"
	@echo "Poetry version: $$(poetry --version)"
	@echo "Python version: $$(poetry run python --version)"
	@echo "Virtual env: $$(poetry env info --path)"
	@echo "Dependencies:"
	@poetry show --only=main

# Platform-specific commands
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    SHELL_EXTENSION := .bat
else
    DETECTED_OS := $(shell uname -s)
    SHELL_EXTENSION := .sh
endif

install-poetry-windows: ## Install Poetry on Windows
	@powershell -Command "(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -"

install-poetry-unix: ## Install Poetry on Unix/Linux/macOS
	@curl -sSL https://install.python-poetry.org | python3 -

# Quick commands for common tasks
start: run ## Alias for run
go: run ## Alias for run
launch: run ## Alias for run
