# Makefile for SigmaPilot

.PHONY: test test-ts test-py test-e2e test-unit test-coverage build install clean help
.PHONY: up down restart rebuild rebuild-clean logs ps wipe init

# Default target
help:
	@echo "SigmaPilot Development Commands"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests (TypeScript + Python)"
	@echo "  make test-ts       Run TypeScript unit tests only"
	@echo "  make test-py       Run Python tests only"
	@echo "  make test-e2e      Run E2E tests (requires dashboard running)"
	@echo "  make test-unit     Run unit tests only (no E2E)"
	@echo "  make test-coverage Run tests with coverage report"
	@echo ""
	@echo "Docker:"
	@echo "  make up            Start all services"
	@echo "  make down          Stop all services"
	@echo "  make restart       Restart all services"
	@echo "  make rebuild       Rebuild containers and restart"
	@echo "  make rebuild-clean Rebuild containers without cache and restart"
	@echo "  make wipe          Stop, remove volumes, rebuild, and start fresh"
	@echo "  make logs          Follow service logs"
	@echo "  make ps            Show service status"
	@echo "  make init          Initialize Alpha Pool with historical data"
	@echo ""
	@echo "Development:"
	@echo "  make build         Build TypeScript"
	@echo "  make install       Install dependencies"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Prerequisites for E2E tests:"
	@echo "  - Dashboard must be running: make up"
	@echo "  - Browser installed: npx playwright install chromium"

#
# Testing
#

# Run all tests
test: test-ts test-py
	@echo ""
	@echo "============================================"
	@echo "All tests completed!"
	@echo "============================================"

# Run TypeScript tests only (excludes E2E by default in npm test)
test-ts:
	@echo "Running TypeScript tests..."
	npm test

# Run Python tests only
test-py:
	@echo "Running Python tests..."
	cd services/hl-decide && python -m pytest -v

# Run E2E tests (requires dashboard running)
test-e2e:
	@echo "Running E2E tests..."
	@echo "Note: Dashboard must be running (make up)"
	npm run test:e2e

# Run unit tests only (TypeScript + Python, no E2E)
test-unit: test-ts test-py
	@echo ""
	@echo "Unit tests completed!"

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	npm run test:coverage
	cd services/hl-decide && python -m pytest --cov=app --cov-report=term-missing

#
# Docker
#

# Start all services
up:
	docker compose up -d

# Stop all services
down:
	docker compose down

# Restart all services
restart:
	docker compose restart

# Rebuild containers and restart
rebuild:
	docker compose down
	docker compose build
	docker compose up -d

# Rebuild containers without cache and restart
rebuild-clean:
	docker compose down
	docker compose build --no-cache
	docker compose up -d

# Wipe everything and start fresh (removes volumes/data)
wipe:
	@echo "WARNING: This will delete all data including database!"
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	docker compose down -v
	docker compose build --no-cache
	docker compose up -d
	@echo "Fresh environment started. Database will be re-initialized."
	@echo "Run 'make init' to populate the Alpha Pool."

# Initialize Alpha Pool with historical data (run after fresh install)
# Uses npm script which works cross-platform (Windows, Mac, Linux)
init:
	@echo "Initializing Alpha Pool with historical data..."
	@echo "This may take several minutes depending on the number of traders."
	npm run init:alpha-pool

# Follow logs
logs:
	docker compose logs -f

# Show service status
ps:
	docker compose ps

#
# Development
#

# Build TypeScript
build:
	npm run build

# Install dependencies
install:
	npm install
	cd services/hl-decide && pip install -r requirements.txt 2>/dev/null || pip install pytest pytest-asyncio pytest-cov pytest-mock faker

# Clean build artifacts
clean:
	rm -rf dist/
	rm -rf packages/ts-lib/dist/
	rm -rf services/hl-scout/dist/
	rm -rf services/hl-stream/dist/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
