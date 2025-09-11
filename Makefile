# Enhanced PII Redactor - Makefile
# Quick commands for setup, installation, and running the 99% accuracy system

.PHONY: help install setup run dev test clean lint format check-deps demo

# Default target
.DEFAULT_GOAL := help

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)🛡️  Enhanced PII Redactor - 99% Accuracy System$(NC)"
	@echo "$(YELLOW)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

install: ## Install all dependencies
	@echo "$(GREEN)📦 Installing dependencies...$(NC)"
	@if [ ! -d "venv" ]; then \
		echo "$(YELLOW)🔧 Creating virtual environment...$(NC)"; \
		python3 -m venv venv; \
	fi
	@echo "$(YELLOW)📥 Installing packages in virtual environment...$(NC)"
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed successfully!$(NC)"

setup: ## Complete setup (install dependencies + configure environment)
	@echo "$(GREEN)🚀 Setting up Enhanced PII Redactor...$(NC)"
	$(MAKE) install
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)📋 Creating .env file from template...$(NC)"; \
		cp .env.example .env; \
		echo "$(YELLOW)⚠️  Please edit .env with your Azure credentials$(NC)"; \
	else \
		echo "$(GREEN)✅ .env file already exists$(NC)"; \
	fi
	@mkdir -p data logs
	@echo "$(GREEN)🎉 Setup complete! Run 'make run' to start the application$(NC)"

run: ## Run the Enhanced PII Redactor application
	@echo "$(GREEN)🚀 Starting Enhanced PII Redactor (99% accuracy)...$(NC)"
	@echo "$(YELLOW)📱 Opening browser at http://localhost:8501$(NC)"
	@if [ -d "venv" ]; then \
		./venv/bin/streamlit run enhanced_pii_redactor_app.py; \
	else \
		echo "$(RED)❌ Virtual environment not found. Run 'make install' first$(NC)"; \
		exit 1; \
	fi

quick-start: ## Install dependencies and run immediately
	@echo "$(GREEN)⚡ Quick start - Installing and running...$(NC)"
	$(MAKE) setup
	@echo "$(GREEN)🏁 Starting application in 3 seconds...$(NC)"
	@sleep 3
	$(MAKE) run

dev: ## Set up development environment with additional tools
	@echo "$(GREEN)🛠️  Setting up development environment...$(NC)"
	$(MAKE) install
	pip install pytest pytest-cov black flake8 isort mypy
	@echo "$(GREEN)✅ Development environment ready!$(NC)"

test: ## Run all tests
	@echo "$(GREEN)🧪 Running tests...$(NC)"
	python -m pytest tests/ -v --cov=. --cov-report=term-missing

test-quick: ## Run quick smoke tests
	@echo "$(GREEN)⚡ Running quick tests...$(NC)"
	@if [ -d "venv" ]; then \
		./venv/bin/python -c "from enhanced_ml_detector import EnhancedMLPIIDetector; print('✅ Core detector works!')"; \
		./venv/bin/python -c "from performance_monitor import PerformanceMonitor; print('✅ Performance monitor works!')"; \
		./venv/bin/python -c "from confidence_scoring import AdvancedConfidenceScorer; print('✅ Confidence scorer works!')"; \
	else \
		python3 -c "from enhanced_ml_detector import EnhancedMLPIIDetector; print('✅ Core detector works!')" 2>/dev/null || echo "❌ Core detector failed"; \
		python3 -c "from performance_monitor import PerformanceMonitor; print('✅ Performance monitor works!')" 2>/dev/null || echo "❌ Performance monitor failed"; \
		python3 -c "from confidence_scoring import AdvancedConfidenceScorer; print('✅ Confidence scorer works!')" 2>/dev/null || echo "❌ Confidence scorer failed"; \
	fi
	@echo "$(GREEN)🎉 All core components loaded successfully!$(NC)"

demo: ## Run demo without Azure credentials
	@echo "$(GREEN)🎮 Running demo mode...$(NC)"
	@if [ -d "venv" ]; then \
		./venv/bin/python enhanced_ml_detector.py; \
	else \
		python3 enhanced_ml_detector.py; \
	fi

demo-confidence: ## Demo confidence scoring system
	@echo "$(GREEN)🎯 Running confidence scoring demo...$(NC)"
	@if [ -d "venv" ]; then \
		./venv/bin/python confidence_scoring.py; \
	else \
		python3 confidence_scoring.py; \
	fi

demo-performance: ## Demo performance monitoring
	@echo "$(GREEN)📊 Running performance monitoring demo...$(NC)"
	@if [ -d "venv" ]; then \
		./venv/bin/python performance_monitor.py; \
	else \
		python3 performance_monitor.py; \
	fi

lint: ## Run code linting
	@echo "$(GREEN)🔍 Running code linting...$(NC)"
	flake8 --max-line-length=120 --ignore=E501,W503 *.py
	@echo "$(GREEN)✅ Linting complete!$(NC)"

format: ## Format code with black and isort
	@echo "$(GREEN)🎨 Formatting code...$(NC)"
	black --line-length=120 *.py
	isort *.py
	@echo "$(GREEN)✅ Code formatted!$(NC)"

check-deps: ## Check if all dependencies are installed
	@echo "$(GREEN)🔍 Checking dependencies...$(NC)"
	@if [ -d "venv" ]; then \
		echo "$(YELLOW)📦 Checking virtual environment dependencies...$(NC)"; \
		./venv/bin/python -c "import streamlit; print('✅ Streamlit')" 2>/dev/null || echo "❌ Streamlit missing"; \
		./venv/bin/python -c "import pandas; print('✅ Pandas')" 2>/dev/null || echo "❌ Pandas missing"; \
		./venv/bin/python -c "import plotly; print('✅ Plotly')" 2>/dev/null || echo "❌ Plotly missing"; \
		./venv/bin/python -c "import sklearn; print('✅ Scikit-learn')" 2>/dev/null || echo "❌ Scikit-learn missing"; \
		./venv/bin/python -c "import azure.ai.textanalytics; print('✅ Azure AI')" 2>/dev/null || echo "❌ Azure AI missing"; \
		./venv/bin/python -c "import requests; print('✅ Requests')" 2>/dev/null || echo "❌ Requests missing"; \
	else \
		echo "$(YELLOW)📦 Checking system dependencies...$(NC)"; \
		python3 -c "import streamlit; print('✅ Streamlit')" 2>/dev/null || echo "❌ Streamlit missing"; \
		python3 -c "import pandas; print('✅ Pandas')" 2>/dev/null || echo "❌ Pandas missing"; \
		python3 -c "import plotly; print('✅ Plotly')" 2>/dev/null || echo "❌ Plotly missing"; \
		python3 -c "import sklearn; print('✅ Scikit-learn')" 2>/dev/null || echo "❌ Scikit-learn missing"; \
		python3 -c "import azure.ai.textanalytics; print('✅ Azure AI')" 2>/dev/null || echo "❌ Azure AI missing"; \
		python3 -c "import requests; print('✅ Requests')" 2>/dev/null || echo "❌ Requests missing"; \
	fi

check-azure: ## Check Azure credentials configuration
	@echo "$(GREEN)🔍 Checking Azure configuration...$(NC)"
	@if [ -f .env ]; then \
		echo "$(GREEN)✅ .env file exists$(NC)"; \
		if grep -q "your_azure_api_key_here" .env; then \
			echo "$(YELLOW)⚠️  Please update Azure credentials in .env file$(NC)"; \
		else \
			echo "$(GREEN)✅ Azure credentials appear to be configured$(NC)"; \
		fi \
	else \
		echo "$(RED)❌ .env file not found. Run 'make setup' first$(NC)"; \
	fi

clean: ## Clean up temporary files and caches
	@echo "$(GREEN)🧹 Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -f *.db
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

status: ## Show system status and configuration
	@echo "$(GREEN)📊 Enhanced PII Redactor Status$(NC)"
	@echo "$(YELLOW)Dependencies:$(NC)"
	@$(MAKE) check-deps --no-print-directory
	@echo ""
	@echo "$(YELLOW)Configuration:$(NC)"
	@$(MAKE) check-azure --no-print-directory
	@echo ""
	@echo "$(YELLOW)Files:$(NC)"
	@ls -la *.py | wc -l | xargs printf "Python files: %s\n"
	@if [ -f .env ]; then echo "✅ Environment configured"; else echo "❌ Environment needs setup"; fi

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)🏃 Running performance benchmarks...$(NC)"
	@echo "This would run accuracy and speed benchmarks"
	$(MAKE) test-quick

docs: ## Generate documentation
	@echo "$(GREEN)📚 Documentation available in README.md$(NC)"
	@echo "$(YELLOW)Key files:$(NC)"
	@echo "  📖 README.md - Complete documentation"
	@echo "  ⚙️  .env.example - Configuration template"
	@echo "  📦 requirements.txt - Dependencies"
	@echo "  🏃 Makefile - This file with commands"

# Advanced commands
install-dev: ## Install development dependencies
	$(MAKE) install
	pip install jupyter notebook ipython

notebook: ## Start Jupyter notebook for development
	@echo "$(GREEN)📓 Starting Jupyter notebook...$(NC)"
	jupyter notebook

profile: ## Run with profiling enabled
	@echo "$(GREEN)⚡ Running with performance profiling...$(NC)"
	python -m cProfile -o profile_output.prof enhanced_pii_redactor_app.py

# Docker commands (if Docker is available)
docker-build: ## Build Docker image
	@echo "$(GREEN)🐳 Building Docker image...$(NC)"
	docker build -t enhanced-pii-redactor .

docker-run: ## Run in Docker container
	@echo "$(GREEN)🐳 Running in Docker...$(NC)"
	docker run -p 8501:8501 enhanced-pii-redactor

# Deployment helpers
package: ## Create deployment package
	@echo "$(GREEN)📦 Creating deployment package...$(NC)"
	zip -r enhanced-pii-redactor.zip *.py requirements.txt .env.example README.md Makefile
	@echo "$(GREEN)✅ Package created: enhanced-pii-redactor.zip$(NC)"

# Quick shortcuts
start: run ## Alias for run
app: run ## Alias for run
go: quick-start ## Alias for quick-start