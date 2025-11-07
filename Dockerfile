# FinBugs Analytics Platform - Production Dockerfile
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy application code
COPY . .

# Compile C++ services
RUN cd cpp_services/monte_carlo && \
    g++ -std=c++17 -O3 -shared -fPIC monte_carlo.cpp -o libmontecarlo.so

RUN cd cpp_services/pair_trading && \
    g++ -std=c++17 -O3 backtest.cpp -o backtest

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
