.PHONY: help install dev build build-dev build-prod clean test lint format deps

# Default target
help: ## Show this help message
	@echo "Redactify - Desktop PII Redactor"
	@echo "================================="
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development
install: ## Install all dependencies (Node.js, Go)
	@echo "Installing dependencies..."
	npm install
	cd frontend && npm install
	cd backend && go mod tidy
	@echo "✅ Dependencies installed"

dev: ## Start development environment
	@echo "Starting development environment..."
	@echo "Starting PostgreSQL..."
	docker-compose up -d postgres
	@echo "Waiting for PostgreSQL to be ready..."
	npx wait-on tcp:5432
	npx concurrently "make dev-backend" "make dev-frontend" "wait-on --timeout 60000 --interval 2000 http://127.0.0.1:3000 http://127.0.0.1:8080/health && electron ."

dev-frontend: ## Start frontend development server
	cd frontend && npm start

dev-backend: ## Start backend development server
	cd backend && go mod tidy && DB_HOST=127.0.0.1 go run cmd/server/main.go

# Building
build-dev: ## Build for development
	@echo "Building for development..."
	npm run build

build-prod: ## Build for production (single executable)
	@echo "Building for production..."
	npm run build:all
	@echo "✅ Production build complete in dist/"

build: build-prod ## Alias for build-prod

# Testing
test: ## Run all tests
	@echo "Running tests..."
	npm run test

test-frontend: ## Run frontend tests only
	cd frontend && npm test

test-backend: ## Run backend tests only
	cd backend && go test ./...

# Code quality
lint: ## Run linters for all code
	@echo "Running linters..."
	npm run lint

lint-frontend: ## Lint frontend code
	cd frontend && npm run lint

lint-backend: ## Lint backend code
	cd backend && golangci-lint run

format: ## Format all code
	@echo "Formatting code..."
	cd frontend && npm run format
	cd backend && gofmt -w .
	cd backend && goimports -w .

# Dependencies
deps-frontend: ## Install frontend dependencies
	cd frontend && npm install

deps-backend: ## Install backend dependencies
	cd backend && go mod tidy && go mod download


# Cleanup
clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	npm run clean
	rm -rf node_modules/
	rm -rf frontend/node_modules/
	rm -rf frontend/build/
	rm -rf backend/redactify-server*
	rm -rf dist/

# Development utilities



# Production deployment
package-win: ## Package for Windows
	npm run build:all
	electron-builder --win

package-mac: ## Package for macOS
	npm run build:all
	electron-builder --mac

package-linux: ## Package for Linux
	npm run build:all
	electron-builder --linux

package-all: ## Package for all platforms
	npm run build:all
	electron-builder --win --mac --linux

# Debug and info
info: ## Show environment information
	@echo "Environment Information:"
	@echo "======================="
	@echo "Node.js: $$(node --version)"
	@echo "npm: $$(npm --version)"
	@echo "Go: $$(go version)"
	@echo "Platform: $$(uname -s)"

check-deps: ## Check if required dependencies are installed
	@echo "Checking dependencies..."
	@command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required"; exit 1; }
	@command -v go >/dev/null 2>&1 || { echo "❌ Go is required"; exit 1; }
	@echo "✅ All dependencies available"

# Quick development setup
setup: check-deps install ## Complete setup for new development environment
	@echo "🚀 Redactify setup complete! Run 'make dev' to start development."