# FinBugs Analytics Platform

[![CI/CD](https://github.com/finbugs/analytics/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/finbugs/analytics/actions)
[![codecov](https://codecov.io/gh/finbugs/analytics/branch/main/graph/badge.svg)](https://codecov.io/gh/finbugs/analytics)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive financial analytics platform providing real-time market sentiment analysis, AI-powered portfolio optimization, predictive modeling, and quantitative trading strategies.

## 🚀 Features

### Market Sentiment Analyzer
- Real-time news sentiment analysis using VADER
- Multi-source data aggregation (Yahoo Finance, NewsAPI)
- Keyword extraction and trend visualization
- Word cloud generation for market narratives

### AI Portfolio Optimizer
- Modern Portfolio Theory (MPT) implementation
- Efficient frontier calculation with Monte Carlo simulation
- Risk metrics: VaR, Sharpe Ratio, Beta
- ML-based asset clustering using K-Means

### Stock Price Predictor
- Multiple ML models: Linear Regression, Random Forest, LSTM
- Technical indicators: RSI, MACD, Bollinger Bands
- Time series forecasting
- Automated model selection based on performance

### Quantitative Trading Services (C++)
- High-performance Monte Carlo simulations
- GARCH volatility modeling
- Jump-diffusion processes
- Pairs trading strategy backtesting

## 📋 Prerequisites

- Python 3.9 or higher
- C++ compiler with C++17 support (g++ or clang++)
- Docker (optional, for containerized deployment)
- PostgreSQL 15+ (for production)
- Redis (for caching and task queue)

## 🛠️ Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/finbugs/analytics.git
cd analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Build C++ services
cd cpp_services/monte_carlo
g++ -std=c++17 -O3 -shared -fPIC monte_carlo.cpp -o libmontecarlo.so
cd ../pair_trading
g++ -std=c++17 -O3 backtest.cpp -o backtest
cd ../..
```

### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🚦 Quick Start

### Market Sentiment Analysis

```python
from market_sentiment.sentiment_analyzer import MarketSentiment

# Initialize analyzer
analyzer = MarketSentiment()

# Analyze a specific ticker
results = analyzer.analyze_ticker("AAPL")

# View sentiment distribution
print(results['sentiment_summary'])
print(results['top_keywords'])
```

### Portfolio Optimization

```python
from portfolio.optimizer import PortfolioOptimizer

# Define portfolio
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
optimizer = PortfolioOptimizer(tickers)

# Calculate optimal weights
optimal_portfolio = optimizer.optimize()

# View metrics
print(f"Expected Return: {optimal_portfolio['return']:.2%}")
print(f"Volatility: {optimal_portfolio['volatility']:.2%}")
print(f"Sharpe Ratio: {optimal_portfolio['sharpe']:.2f}")
```

### Stock Price Prediction

```python
from stock_price_predictor.predictor import StockPredictor

# Initialize predictor
predictor = StockPredictor("TSLA")

# Train models
predictor.train_models()

# Predict future prices
future_predictions = predictor.predict_future(days=30)
print(future_predictions)
```

### Monte Carlo Simulation (C++)

```bash
cd cpp_services/monte_carlo
./monte_carlo --ticker AAPL --simulations 10000 --days 252
```

### Pair Trading Backtest (C++)

```bash
cd cpp_services/pair_trading
./backtest --pair1 AAPL --pair2 MSFT --window 20
```

## 📊 API Documentation

### REST API

The platform exposes a RESTful API for all services:

```bash
# Start API server
uvicorn api.main:app --reload

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

**Key Endpoints:**

- `GET /api/v1/sentiment/{ticker}` - Get sentiment analysis
- `POST /api/v1/portfolio/optimize` - Optimize portfolio
- `GET /api/v1/predict/{ticker}` - Get price predictions
- `POST /api/v1/monte-carlo/simulate` - Run Monte Carlo simulation
- `GET /api/v1/health` - Health check

## 🏗️ Architecture

```
finbugs/
├── market_sentiment/          # Sentiment analysis module
│   └── sentiment_analyzer.py
├── portfolio/                 # Portfolio optimization
│   └── optimizer.py
├── stock_price_predictor/    # ML prediction models
│   └── predictor.py
├── cpp_services/             # High-performance C++ services
│   ├── monte_carlo/
│   └── pair_trading/
├── api/                      # REST API
│   └── main.py
├── shared/                   # Shared utilities
├── data/                     # Data storage
├── tests/                    # Test suite
├── config/                   # Configuration files
└── docs/                     # Documentation

```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific module tests
pytest tests/test_sentiment.py
pytest tests/test_portfolio.py
pytest tests/test_predictor.py

# C++ tests
cd cpp_services/monte_carlo && make test
cd cpp_services/pair_trading && make test
```

## 📈 Performance

- **Sentiment Analysis**: ~200 headlines/second
- **Portfolio Optimization**: 5000 simulations in ~2 seconds
- **Monte Carlo**: 10,000 simulations in ~500ms (C++)
- **Pair Trading Backtest**: 5 years of daily data in ~100ms

## 🔧 Configuration

Key configuration options in `.env`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DATABASE_URL=postgresql://user:pass@localhost/finbugs

# External APIs
YAHOO_FINANCE_TIMEOUT=30
NEWS_API_KEY=your_api_key_here

# Model Parameters
MC_NUM_SIMULATIONS=10000
PORTFOLIO_NUM_PORTFOLIOS=5000
SENTIMENT_VADER_THRESHOLD=0.05
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Python: Follow PEP 8, use Black formatter
- C++: Follow Google C++ Style Guide
- Write unit tests for new features
- Update documentation as needed

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Engineering Team**: [team@finbugs.io](mailto:team@finbugs.io)
- **Support**: [support@finbugs.io](mailto:support@finbugs.io)

## 🙏 Acknowledgments

- VADER Sentiment Analysis
- Yahoo Finance API
- scikit-learn
- TensorFlow/Keras
- Modern Portfolio Theory research

## 📚 Resources

- [Full Documentation](https://docs.finbugs.io)
- [API Reference](https://api.finbugs.io/docs)
- [Blog](https://blog.finbugs.io)
- [Community Forum](https://community.finbugs.io)

## ⚠️ Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions.

## 🐛 Bug Reports

Found a bug? Please report it:

1. Check existing issues first
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details

## 📊 Monitoring

Production deployments include:

- Prometheus metrics at `:9090/metrics`
- Health checks at `/health`
- Logging to `logs/` directory
- Sentry error tracking (configure with SENTRY_DSN)

---

**Built with ❤️ by the FinBugs Team**
