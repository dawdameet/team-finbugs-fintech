import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta

class StockPredictor:
    """
    Multi-model stock price predictor using various ML techniques
    Models included: LSTM, Linear Regression, Random Forest, ARIMA
    """
    
    def __init__(self, ticker, period='2y'):
        """
        Initialize the predictor with stock data
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period ('1y', '2y', '5y', 'max')
        """
        self.ticker = ticker
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {}
        self.predictions = {}
        
        print(f"Fetching data for {ticker}...")
        self.fetch_data(period)
        
    def fetch_data(self, period):
        """Fetch stock data from Yahoo Finance"""
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=period)
        print(f"Data fetched: {len(self.data)} days")
        
    def create_features(self, df):
        """
        Create technical indicators as features
        """
        df = df.copy()
        
        # Moving Averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Price change
        df['Price_Change'] = df['Close'].pct_change()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        
        # --- BUG [HARD] ---
        # Incorrect 'loss' calculation for RSI. It's missing the negative sign, 
        # so 'loss' will be a negative value, corrupting the 'rs' and 'RSI' values.
        loss = (delta.where(delta < 0, 0)).rolling(window=14).mean() # <-- BUG
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        return df.dropna()
    
    def prepare_lstm_data(self, lookback=60):
        """
        Prepare data for LSTM model
        
        Args:
            lookback: Number of previous days to use for prediction
        """
        df = self.create_features(self.data)
        data = df['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        split = int(0.8 * len(X))
        return X[:split], X[split:], y[:split], y[split:]
    
    def prepare_ml_data(self):
        """Prepare data for traditional ML models"""
        df = self.create_features(self.data)
        
        feature_cols = ['Open', 'High', 'Low', 'Volume', 'MA7', 'MA21', 
                       'Volatility', 'Price_Change', 'RSI', 'MACD']
        
        df = df.dropna()
        X = df[feature_cols].values
        y = df['Close'].values
        
        # --- BUG [MEDIUM] ---
        # Data is being shuffled (shuffle=True by default).
        # For time-series data, this is critical data leakage, as the model
        # will be trained on future data to predict the past, resulting in
        # unrealistically high R² scores. `shuffle=False` is required.
        return train_test_split(X, y, test_size=0.2,shuffle=False) # <-- BUG
    
    def train_lstm(self, epochs=50, batch_size=32):
        """
        Train LSTM model (requires tensorflow/keras)
        Note: Install with: pip install tensorflow
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            print("\n--- Training LSTM Model ---")
            X_train, X_test, y_train, y_test = self.prepare_lstm_data()
            
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50),
                Dropout(0.2),
                Dense(units=1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                     validation_data=(X_test, y_test), verbose=0)
            
            predictions = model.predict(X_test)
            predictions = self.scaler.inverse_transform(predictions)
            y_test_scaled = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            self.models['LSTM'] = model
            self.predictions['LSTM'] = {
                'pred': predictions.flatten(),
                'actual': y_test_scaled.flatten(),
                'mse': mean_squared_error(y_test_scaled, predictions),
                'mae': mean_absolute_error(y_test_scaled, predictions),
                'r2': r2_score(y_test_scaled, predictions)
            }
            
            print(f"LSTM - MSE: {self.predictions['LSTM']['mse']:.2f}, "
                  f"MAE: {self.predictions['LSTM']['mae']:.2f}, "
                  f"R²: {self.predictions['LSTM']['r2']:.4f}")
            
        except ImportError:
            print("TensorFlow not installed. Skipping LSTM model.")
            print("Install with: pip install tensorflow")
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        print("\n--- Training Linear Regression ---")
        X_train, X_test, y_train, y_test = self.prepare_ml_data()
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        self.models['Linear'] = model
        self.predictions['Linear'] = {
            'pred': predictions,
            'actual': y_test,
            'mse': mean_squared_error(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        print(f"Linear - MSE: {self.predictions['Linear']['mse']:.2f}, "
              f"MAE: {self.predictions['Linear']['mae']:.2f}, "
              f"R²: {self.predictions['Linear']['r2']:.4f}")
    
    def train_random_forest(self, n_estimators=100):
        """Train Random Forest model"""
        print("\n--- Training Random Forest ---")
        X_train, X_test, y_train, y_test = self.prepare_ml_data()
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        self.models['RandomForest'] = model
        self.predictions['RandomForest'] = {
            'pred': predictions,
            'actual': y_test,
            'mse': mean_squared_error(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions)
        }
        
        print(f"Random Forest - MSE: {self.predictions['RandomForest']['mse']:.2f}, "
              f"MAE: {self.predictions['RandomForest']['mae']:.2f}, "
              f"R²: {self.predictions['RandomForest']['r2']:.4f}")
    
    def train_arima(self):
        """
        Train ARIMA model (requires statsmodels)
        Note: Install with: pip install statsmodels
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            print("\n--- Training ARIMA Model ---")
            df = self.data['Close'].dropna()
            
            # Split data
            split = int(0.8 * len(df))
            train, test = df[:split], df[split:]
            
            # Fit ARIMA model
            model = ARIMA(train, order=(5, 1, 0))
            fitted_model = model.fit()
            
            # Make predictions
            predictions = fitted_model.forecast(steps=len(test))
            
            self.models['ARIMA'] = fitted_model
            self.predictions['ARIMA'] = {
                'pred': predictions.values,
                'actual': test.values,
                'mse': mean_squared_error(test, predictions),
                'mae': mean_absolute_error(test, predictions),
                'r2': r2_score(test, predictions)
            }
            
            print(f"ARIMA - MSE: {self.predictions['ARIMA']['mse']:.2f}, "
                  f"MAE: {self.predictions['ARIMA']['mae']:.2f}, "
                  f"R²: {self.predictions['ARIMA']['r2']:.4f}")
            
        except ImportError:
            print("Statsmodels not installed. Skipping ARIMA model.")
            print("Install with: pip install statsmodels")
    
    def train_all(self):
        """Train all available models"""
        self.train_linear_regression()
        self.train_random_forest()
        self.train_arima()
        self.train_lstm()
    
    def plot_predictions(self):
        """Visualize predictions from all models"""
        n_models = len(self.predictions)
        if n_models == 0:
            print("No models trained yet!")
            return
        
        fig, axes = plt.subplots(n_models, 1, figsize=(14, 5*n_models))
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, pred_data) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            actual = pred_data['actual']
            predicted = pred_data['pred']
            
            # --- BUG [EASY] ---
            # This is plotting the predicted values against themselves, so the
            # 'Actual' and 'Predicted' lines will be identical.
            # It should be plotting 'actual' vs 'predicted'.
            ax.plot(actual, label='Actual', linewidth=2, color='blue') # <-- BUG
            ax.plot(predicted, label='Predicted', linewidth=2, color='red', alpha=0.7)
            
            ax.set_title(f'{name} Model - {self.ticker}\n'
                        f'MAE: ${pred_data["mae"]:.2f} | R²: {pred_data["r2"]:.4f}',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Period')
            ax.set_ylabel('Stock Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, days=30):
        """
        Predict future stock prices
        
        Args:
            days: Number of days to predict into the future
        """
        print(f"\n--- Predicting next {days} days ---")
        
        # Use the best performing model (highest R²)
        best_model_name = max(self.predictions, 
                             key=lambda x: self.predictions[x]['r2'])
        
        print(f"Using best model: {best_model_name}")
        print(f"R² Score: {self.predictions[best_model_name]['r2']:.4f}")
        
        # Get the required features for prediction based on best model
        if best_model_name == 'LSTM':
            # For LSTM, we need the last sequence_length days of scaled data
            sequence_length = 60  # Same as lookback in prepare_lstm_data
            last_sequence = self.data['Close'].values[-sequence_length:]
            last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))
            
            future_prices = []
            current_sequence = last_sequence_scaled.copy()
            
            for _ in range(days):
                # Reshape sequence for LSTM input (samples, time steps, features)
                current_input = current_sequence.reshape(1, sequence_length, 1)
                # Get next day prediction
                next_day = self.models[best_model_name].predict(current_input)
                future_prices.append(self.scaler.inverse_transform(next_day.reshape(-1, 1))[0][0])
                # Update sequence: remove oldest, add prediction
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = next_day
        else:
            # For other models (Linear, RandomForest), prepare latest feature set
            df = self.create_features(self.data)
            latest_features = df.iloc[-1:][['Open', 'High', 'Low', 'Volume', 'MA7', 'MA21', 
                                         'Volatility', 'Price_Change', 'RSI', 'MACD']].values
            
            future_prices = []
            current_features = latest_features.copy()
            last_price = self.data['Close'].iloc[-1]
            
            for _ in range(days):
                # Predict next day's price
                next_price = self.models[best_model_name].predict(current_features)[0]
                future_prices.append(next_price)
                
                # Update features for next prediction
                price_change = (next_price - last_price) / last_price
                current_features[0, -2] = price_change  # Update Price_Change feature
                last_price = next_price
        
        future_dates = pd.date_range(
            start=self.data.index[-1] + timedelta(days=1),
            periods=days
        )
        
        plt.figure(figsize=(14, 6))
        plt.plot(self.data.index[-60:], self.data['Close'].iloc[-60:], 
                label='Historical', linewidth=2)
        plt.plot(future_dates, future_prices, 
                label=f'Predicted ({best_model_name})', 
                linewidth=2, linestyle='--', color='red')
        plt.title(f'{self.ticker} - Price Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices
        })


# Example Usage
if __name__ == "__main__":
    # Initialize predictor with a stock ticker
    predictor = StockPredictor('AAPL', period='2y')
    
    # Train all models
    predictor.train_all()
    
    # Visualize predictions
    predictor.plot_predictions()
    
    # Predict future prices
    future_predictions = predictor.predict_future(days=30)
    print("\nFuture Predictions:")
    print(future_predictions.head(10))
    
    # Compare model performance
    print("\n=== Model Comparison ===")
    for name, metrics in predictor.predictions.items():
        print(f"\n{name}:")
        print(f"  MSE: ${metrics['mse']:.2f}")
        print(f"  MAE: ${metrics['mae']:.2f}")
        print(f"  R²:  {metrics['r2']:.4f}")