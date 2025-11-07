from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="finbugs-analytics",
    version="1.0.0",
    author="FinBugs Team",
    author_email="team@finbugs.io",
    description="Comprehensive financial analytics platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/finbugs/analytics",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "finbugs-sentiment=market_sentiment.sentiment_analyzer:main",
            "finbugs-portfolio=portfolio.optimizer:main",
            "finbugs-predictor=stock_price_predictor.predictor:main",
        ],
    },
)
