import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AIPortfolioOptimizer:
    """
    AI-Powered Portfolio Optimizer and Risk Manager
    Features:
    - Modern Portfolio Theory (MPT) optimization
    - Machine Learning for asset clustering
    - Risk-adjusted return maximization
    - Portfolio rebalancing recommendations
    - VaR (Value at Risk) analysis
    - Diversification scoring
    """
    
    def __init__(self, tickers, investment_amount=10000, risk_free_rate=0.02):
        """
        Initialize portfolio optimizer
        
        Args:
            tickers: List of stock tickers
            investment_amount: Total investment amount
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.tickers = tickers
        self.investment_amount = investment_amount
        self.risk_free_rate = risk_free_rate
        self.data = None
        self.returns = None
        self.optimal_weights = None
        
        print(f"Initializing Portfolio Optimizer for {len(tickers)} assets")
        print(f"Investment Amount: ${investment_amount:,.2f}")
        
    def fetch_data(self, period='2y'):
        """Fetch historical data for all tickers"""
        print(f"\nFetching {period} of historical data...")
        
        # Download data
        raw_data = yf.download(self.tickers, period=period, progress=False)
        
        # Handle different return formats
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Multiple tickers - extract 'Adj Close' level
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns.get_level_values(0):
                data = raw_data['Close']
            else:
                raise ValueError("Could not find price data in downloaded data")
        else:
            # Single ticker or already flat
            if 'Adj Close' in raw_data.columns:
                data = raw_data[['Adj Close']]
                data.columns = [self.tickers[0]]
            elif 'Close' in raw_data.columns:
                data = raw_data[['Close']]
                data.columns = [self.tickers[0]]
            else:
                data = raw_data
        
        # Convert Series to DataFrame if needed
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = [self.tickers[0]]
        
        self.data = data.dropna()
        self.returns = self.data.pct_change().dropna()
        
        print(f"Data fetched: {len(self.data)} days")
        print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        
    def calculate_portfolio_metrics(self, weights):
        """
        Calculate portfolio return, volatility, and Sharpe ratio
        
        Args:
            weights: Array of portfolio weights
        """
        # Annual return
        portfolio_return = np.sum(self.returns.mean() * weights) * 252
        
        portfolio_volatility = np.sqrt(
            np.dot(weights.T, np.dot(self.returns.cov(), weights))
        )
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def optimize_portfolio(self, target='sharpe', min_weight=0.01, max_weight=0.4):
        """
        Optimize portfolio using Modern Portfolio Theory
        
        Args:
            target: 'sharpe' (max Sharpe) or 'min_volatility'
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset
        """
        print(f"\nOptimizing portfolio (Target: {target})...")
        
        n_assets = len(self.tickers)
        
        # Objective functions
        def negative_sharpe(weights):
            return -self.calculate_portfolio_metrics(weights)[2]
        
        def portfolio_volatility(weights):
            return self.calculate_portfolio_metrics(weights)[1]
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        if target == 'sharpe':
            result = minimize(negative_sharpe, initial_weights, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            result = minimize(portfolio_volatility, initial_weights,
                            method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.optimal_weights = result.x
        
        # Calculate metrics
        ret, vol, sharpe = self.calculate_portfolio_metrics(self.optimal_weights)
        
        print("\n" + "="*60)
        print("OPTIMAL PORTFOLIO")
        print("="*60)
        
        for ticker, weight in zip(self.tickers, self.optimal_weights):
            allocation = self.investment_amount * weight
            print(f"{ticker:6s}: {weight*100:6.2f}% (${allocation:10,.2f})")
        
        print("-"*60)
        print(f"Expected Annual Return:  {ret*100:6.2f}%")
        print(f"Annual Volatility:       {vol*100:6.2f}%")
        print(f"Sharpe Ratio:            {sharpe:6.2f}")
        print("="*60)
        
        return self.optimal_weights
    
    def generate_efficient_frontier(self, n_portfolios=10000):
        """Generate efficient frontier using Monte Carlo simulation"""
        print("\nGenerating Efficient Frontier...")
        
        n_assets = len(self.tickers)
        results = np.zeros((3, n_portfolios))
        weights_record = []
        
        for i in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= n_assets
            
            weights_record.append(weights)
            
            # Calculate metrics
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights)
            results[0,i] = ret
            results[1,i] = vol
            results[2,i] = sharpe
        
        # Find max Sharpe ratio portfolio
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_return = results[0, max_sharpe_idx]
        max_sharpe_vol = results[1, max_sharpe_idx]
        
        # Find min volatility portfolio
        min_vol_idx = np.argmin(results[1])
        min_vol_return = results[0, min_vol_idx]
        min_vol_vol = results[1, min_vol_idx]
        
        # Plot
        plt.figure(figsize=(14, 8))
        scatter = plt.scatter(results[1,:]*100, results[0,:]*100, 
                            c=results[2,:], cmap='viridis', alpha=0.6, s=10)
        plt.colorbar(scatter, label='Sharpe Ratio')
        
        # Highlight optimal portfolios
        plt.scatter(max_sharpe_vol*100, max_sharpe_return*100, 
                   marker='*', color='red', s=500, 
                   label=f'Max Sharpe ({results[2, max_sharpe_idx]:.2f})')
        plt.scatter(min_vol_vol*100, min_vol_return*100,
                   marker='*', color='blue', s=500,
                   label='Min Volatility')
        
        # Plot optimal portfolio
        if self.optimal_weights is not None:
            opt_ret, opt_vol, opt_sharpe = self.calculate_portfolio_metrics(self.optimal_weights)
            plt.scatter(opt_vol*100, opt_ret*100,
                       marker='X', color='gold', s=500, edgecolors='black',
                       label=f'Optimal Portfolio', zorder=5)
        
        plt.xlabel('Volatility (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
        plt.title('Efficient Frontier - Portfolio Optimization', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return results, weights_record
    
    def ml_asset_clustering(self, n_clusters=3):
        """
        Use Machine Learning to cluster similar assets
        Helps identify diversification opportunities
        """
        print(f"\nPerforming ML Asset Clustering ({n_clusters} clusters)...")
        
        # Feature engineering: statistical properties
        features = pd.DataFrame()
        features['mean_return'] = self.returns.mean()
        features['volatility'] = self.returns.std()
        features['sharpe'] = (features['mean_return'] * 252 - self.risk_free_rate) / (features['volatility'] * np.sqrt(252))
        features['skewness'] = self.returns.skew()
        features['kurtosis'] = self.returns.kurtosis()
        
        market_proxy = self.returns.iloc[:, 0]
        features['beta'] = self.returns.apply(lambda x: x.cov(market_proxy) / x.var())
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # PCA for visualization
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        # Visualize clusters
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Cluster visualization
        scatter = ax1.scatter(features_pca[:, 0], features_pca[:, 1], 
                             c=clusters, cmap='tab10', s=200, alpha=0.7, edgecolors='black')
        for i, ticker in enumerate(self.tickers):
            ax1.annotate(ticker, (features_pca[i, 0], features_pca[i, 1]),
                        fontsize=10, fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
        ax1.set_title('Asset Clustering (PCA)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Cluster')
        
        # Risk-Return plot with clusters
        ax2.scatter(features['volatility']*np.sqrt(252)*100, 
                   features['mean_return']*252*100,
                   c=clusters, cmap='tab10', s=200, alpha=0.7, edgecolors='black')
        for i, ticker in enumerate(self.tickers):
            ax2.annotate(ticker, 
                        (features['volatility'].iloc[i]*np.sqrt(252)*100,
                         features['mean_return'].iloc[i]*252*100),
                        fontsize=10, fontweight='bold')
        ax2.set_xlabel('Annual Volatility (%)', fontweight='bold')
        ax2.set_ylabel('Annual Return (%)', fontweight='bold')
        ax2.set_title('Risk-Return Profile by Cluster', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print cluster analysis
        features['Cluster'] = clusters
        print("\nCluster Analysis:")
        for cluster in range(n_clusters):
            assets = features[features['Cluster'] == cluster].index.tolist()
            print(f"\nCluster {cluster}: {', '.join(assets)}")
            print(f"  Avg Annual Return: {features[features['Cluster']==cluster]['mean_return'].mean()*252*100:.2f}%")
            print(f"  Avg Volatility: {features[features['Cluster']==cluster]['volatility'].mean()*np.sqrt(252)*100:.2f}%")
        
        return features, clusters
    
    def calculate_var(self, confidence_level=0.95, time_horizon=1):
        """
        Calculate Value at Risk (VaR)
        
        Args:
            confidence_level: Confidence level (default: 95%)
            time_horizon: Time horizon in days
        """
        if self.optimal_weights is None:
            print("Optimize portfolio first!")
            return
        
        print(f"\nCalculating VaR (Confidence: {confidence_level*100}%, Horizon: {time_horizon} day(s))")
        
        portfolio_returns = (self.returns * self.optimal_weights).sum(axis=1)
        var_historical = np.percentile(portfolio_returns, confidence_level*100)
        
        var_amount = self.investment_amount * var_historical * np.sqrt(time_horizon)
        
        # Parametric VaR (assumes normal distribution)
        mean = portfolio_returns.mean()
        std = portfolio_returns.std()
        from scipy import stats
        var_parametric = stats.norm.ppf(1-confidence_level, mean, std)
        var_param_amount = self.investment_amount * var_parametric * np.sqrt(time_horizon)
        
        print(f"\nValue at Risk ({confidence_level*100}% confidence):")
        print(f"  Historical VaR: ${abs(var_amount):,.2f} ({var_historical*100:.2f}%)")
        print(f"  Parametric VaR: ${abs(var_param_amount):,.2f} ({var_parametric*100:.2f}%)")
        print(f"\nInterpretation: There is a {(1-confidence_level)*100}% chance of losing more than")
        print(f"${abs(var_amount):,.2f} in {time_horizon} day(s)")
        
        # Plot VaR
        plt.figure(figsize=(12, 6))
        plt.hist(portfolio_returns*100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(var_historical*100, color='red', linestyle='--', linewidth=2,
                   label=f'Historical VaR ({var_historical*100:.2f}%)')
        plt.axvline(var_parametric*100, color='orange', linestyle='--', linewidth=2,
                   label=f'Parametric VaR ({var_parametric*100:.2f}%)')
        plt.xlabel('Daily Returns (%)', fontweight='bold')
        plt.ylabel('Frequency', fontweight='bold')
        plt.title(f'Portfolio Return Distribution & VaR ({confidence_level*100}%)', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return var_historical, var_parametric
    
    def diversification_analysis(self):
        """Analyze portfolio diversification"""
        print("\nAnalyzing Portfolio Diversification...")
        
        # Correlation matrix
        corr_matrix = self.returns.corr()
        
        # Diversification ratio
        if self.optimal_weights is not None:
            weighted_vol = np.sum(self.optimal_weights * self.returns.std() * np.sqrt(252))
            portfolio_vol = self.calculate_portfolio_metrics(self.optimal_weights)[1]
            diversification_ratio = weighted_vol / portfolio_vol
        else:
            diversification_ratio = None
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Asset Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Diversification score (lower avg correlation = better)
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        diversification_score = (1 - avg_correlation) * 100
        
        print(f"\nDiversification Metrics:")
        print(f"  Average Correlation: {avg_correlation:.3f}")
        print(f"  Diversification Score: {diversification_score:.1f}/100")
        if diversification_ratio:
            print(f"  Diversification Ratio: {diversification_ratio:.2f}")
        
        if diversification_score > 70:
            print("  ✅ Excellent diversification!")
        elif diversification_score > 50:
            print("  ⚠️  Moderate diversification")
        else:
            print("  ❌ Poor diversification - consider more uncorrelated assets")
        
        return corr_matrix, diversification_score
    
    def rebalancing_recommendation(self, current_weights):
        """
        Compare current portfolio with optimal and suggest rebalancing
        
        Args:
            current_weights: Dictionary of {ticker: weight}
        """
        if self.optimal_weights is None:
            print("Optimize portfolio first!")
            return
        
        print("\n" + "="*70)
        print("REBALANCING RECOMMENDATIONS")
        print("="*70)
        
        current_weights_array = np.array([current_weights.get(t, 0) for t in self.tickers])
        
        # Calculate current portfolio metrics
        current_ret, current_vol, current_sharpe = self.calculate_portfolio_metrics(current_weights_array)
        optimal_ret, optimal_vol, optimal_sharpe = self.calculate_portfolio_metrics(self.optimal_weights)
        
        print(f"\n{'Asset':<8} {'Current':<12} {'Optimal':<12} {'Change':<12} {'Action'}")
        print("-"*70)
        
        trades = []
        for ticker, curr_w, opt_w in zip(self.tickers, current_weights_array, self.optimal_weights):
            diff = opt_w - curr_w
            curr_amt = self.investment_amount * curr_w
            opt_amt = self.investment_amount * opt_w
            change_amt = self.investment_amount * diff
            
            if abs(diff) > 0.01:  # Only show if change > 1%
                action = "🔴 SELL" if diff < 0 else "🟢 BUY"
                print(f"{ticker:<8} {curr_w*100:>5.2f}% ({curr_amt:>7,.0f}) {opt_w*100:>5.2f}% ({opt_amt:>7,.0f}) "
                      f"{diff*100:>+6.2f}% ({change_amt:>+8,.0f}) {action}")
                trades.append((ticker, change_amt))
        
        print("-"*70)
        print(f"\nCurrent Portfolio:")
        print(f"  Expected Return: {current_ret*100:.2f}%  |  Volatility: {current_vol*100:.2f}%  |  Sharpe: {current_sharpe:.2f}")
        print(f"\nOptimal Portfolio:")
        print(f"  Expected Return: {optimal_ret*100:.2f}%  |  Volatility: {optimal_vol*100:.2f}%  |  Sharpe: {optimal_sharpe:.2f}")
        print(f"\nImprovement:")
        print(f"  Return: {(optimal_ret-current_ret)*100:.2f}%  |  Risk: {(optimal_vol-current_vol)*100:+.2f}%  |  Sharpe: {(optimal_sharpe-current_sharpe):+.2f}")
        print("="*70)
        
        return trades


# Example Usage
if __name__ == "__main__":
    # Define portfolio of diverse assets
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'XOM', 'GLD', 'TLT', 'VNQ']
    
    # Initialize optimizer
    optimizer = AIPortfolioOptimizer(
        tickers=tickers,
        investment_amount=100000,
        risk_free_rate=0.02
    )
    
    # Fetch historical data
    optimizer.fetch_data(period='2y')
    
    # Optimize portfolio for maximum Sharpe ratio
    optimal_weights = optimizer.optimize_portfolio(target='sharpe', max_weight=0.3)
    
    # Generate efficient frontier
    results, weights = optimizer.generate_efficient_frontier(n_portfolios=5000)
    
    # ML-based asset clustering
    features, clusters = optimizer.ml_asset_clustering(n_clusters=3)
    
    # Calculate Value at Risk
    var_hist, var_param = optimizer.calculate_var(confidence_level=0.95, time_horizon=1)
    
    # Diversification analysis
    corr_matrix, div_score = optimizer.diversification_analysis()
    
    # Simulate current portfolio and get rebalancing recommendations
    current_portfolio = {
        'AAPL': 0.15, 'MSFT': 0.15, 'GOOGL': 0.15, 'AMZN': 0.15,
        'JPM': 0.10, 'JNJ': 0.10, 'XOM': 0.05, 'GLD': 0.05,
        'TLT': 0.05, 'VNQ': 0.05
    }
    trades = optimizer.rebalancing_recommendation(current_portfolio)