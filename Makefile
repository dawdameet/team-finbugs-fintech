.PHONY: help install test lint format clean build-cpp run docker-build docker-up docker-down

# Default target
help:
	@echo "FinBugs Analytics Platform - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install Python dependencies"
	@echo "  make build-cpp      - Compile C++ services"
	@echo "  make setup          - Complete setup (install + build-cpp)"
	@echo ""
	@echo "Development:"
	@echo "  make run            - Start development server"
	@echo "  make test           - Run all tests"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code with Black"
	@echo "  make clean          - Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-up      - Start Docker services"
	@echo "  make docker-down    - Stop Docker services"
	@echo ""

# Installation
install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Build C++ services
build-cpp:
	@echo "Building Monte Carlo service..."
	cd cpp_services/monte_carlo && \
		g++ -std=c++17 -O3 -Wall -Wextra -shared -fPIC monte_carlo.cpp -o libmontecarlo.so
	@echo "Building Pair Trading service..."
	cd cpp_services/pair_trading && \
		g++ -std=c++17 -O3 -Wall -Wextra backtest.cpp -o backtest
	@echo "C++ services built successfully!"

# Complete setup
setup: install build-cpp
	@echo "Setup complete!"

# Run development server
run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Testing
test:
	pytest tests/ -v --cov=. --cov-report=term-missing

test-quick:
	pytest tests/ -v

# Linting
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy market_sentiment portfolio stock_price_predictor --ignore-missing-imports

# Formatting
format:
	black .
	isort .

# Clean build artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.so" -delete
	find . -type f -name "*.o" -delete
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	cd cpp_services/monte_carlo && rm -f libmontecarlo.so monte_carlo
	cd cpp_services/pair_trading && rm -f backtest

# Docker commands
docker-build:
	docker build -t finbugs:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Database migrations (if using Alembic)
db-migrate:
	alembic upgrade head

db-rollback:
	alembic downgrade -1

# Run sentiment analysis
run-sentiment:
	python -m market_sentiment.sentiment_analyzer

# Run portfolio optimizer
run-portfolio:
	python -m portfolio.optimizer

# Run stock predictor
run-predictor:
	python -m stock_price_predictor.predictor

# Run Monte Carlo
run-monte-carlo:
	cd cpp_services/monte_carlo && ./monte_carlo

# Run pair trading backtest
run-pairs:
	cd cpp_services/pair_trading && ./backtest
