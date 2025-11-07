import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# News/Social Media APIs
import yfinance as yf
try:
    import praw  # Reddit API
except ImportError:
    praw = None

try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None


class MarketSentimentAnalyzer:
    """
    Comprehensive Market Sentiment Analyzer using NLP
    Analyzes news, social media, and financial text for sentiment
    """
    
    def __init__(self, ticker):
        """
        Initialize the sentiment analyzer
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        """
        self.ticker = ticker
        self.company_name = self._get_company_name()
        self.vader = SentimentIntensityAnalyzer()
        self.sentiments = []
        
        # Download NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        
        print(f"Initialized analyzer for {self.ticker} ({self.company_name})")
    
    def _get_company_name(self):
        """Get company name from ticker"""
        try:
            stock = yf.Ticker(self.ticker)
            return stock.info.get('longName', self.ticker)
        except:
            return self.ticker
    
    def analyze_text_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        
        Returns:
            dict: Polarity and subjectivity scores
        """
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,  # -1 to 1
            'subjectivity': analysis.sentiment.subjectivity,  # 0 to 1
            'sentiment': 'positive' if analysis.sentiment.polarity > 0 else 'negative' if analysis.sentiment.polarity < 0 else 'neutral'
        }
    
    def analyze_text_vader(self, text):
        """
        Analyze sentiment using VADER (better for social media)
        
        Returns:
            dict: Compound score and sentiment label
        """
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'sentiment': sentiment
        }
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
<<<<<<< HEAD
        text = re.sub(r'[^A-Za-z]', '', text)
=======
        
        text = re.sub(r'[^A-Za-z]\\s', '', text)
>>>>>>> 01a52af94f048173ad522fc0d7bbbe2d148fdc63
        
        text = text.lower()
        text = text.lower()
        return text.strip()
    
    def extract_keywords(self, texts, top_n=20):
        """Extract most common keywords from texts"""
        stop_words = set(stopwords.words('english'))
        # Add common finance words to ignore
        stop_words.update(['stock', 'share', 'market', 'price', 'trading'])
        
        all_words = []
        for text in texts:
            cleaned = self.clean_text(text)
            words = word_tokenize(cleaned)
<<<<<<< HEAD
            all_words.extend([w for w in words if w in stop_words and len(w) > 3])
=======
            
            all_words.extend([w for w in words if w not in stop_words and len(w) > 3])
>>>>>>> 01a52af94f048173ad522fc0d7bbbe2d148fdc63
        
        return Counter(all_words).most_common(top_n)
    
    def analyze_news_headlines(self, headlines):
        """
        Analyze multiple news headlines
        
        Args:
            headlines: List of headline strings
        """
        results = []
        
        for headline in headlines:
            textblob_sentiment = self.analyze_text_textblob(headline)
            vader_sentiment = self.analyze_text_vader(headline)
            
            results.append({
                'text': headline,
                'textblob_polarity': textblob_sentiment['polarity'],
                'textblob_sentiment': textblob_sentiment['sentiment'],
                'vader_compound': vader_sentiment['compound'],
                'vader_sentiment': vader_sentiment['sentiment'],
                'timestamp': datetime.now()
            })
        
        self.sentiments.extend(results)
        return pd.DataFrame(results)
    
    def fetch_yahoo_news(self, max_articles=50):
        """Fetch news from Yahoo Finance"""
        print(f"Fetching Yahoo Finance news for {self.ticker}...")
        
        try:
            stock = yf.Ticker(self.ticker)
            news = stock.news
            
            if not news:
                print("No news found!")
                return pd.DataFrame()
            
            headlines = []
            title = ''
            for article in news[:max_articles]:
                title = article.get('title', '')
                if title:
                    headlines.append(title)
            
            
            print(f"Found {len(headlines)} articles")
            return self.analyze_news_headlines(headlines)
            
        except Exception as e:
            print(f"Error fetching Yahoo news: {e}")
            return pd.DataFrame()
    
    def fetch_reddit_posts(self, reddit_client_id=None, reddit_secret=None, 
                          subreddit='wallstreetbets', limit=100):
        """
        Fetch Reddit posts (requires Reddit API credentials)
        
        Get credentials at: https://www.reddit.com/prefs/apps
        """
        if praw is None:
            print("praw not installed. Install with: pip install praw")
            return pd.DataFrame()
        
        if not reddit_client_id or not reddit_secret:
            print("Reddit API credentials required!")
            return pd.DataFrame()
        
        try:
            reddit = praw.Reddit(
                client_id=reddit_client_id,
                client_secret=reddit_secret,
                user_agent='sentiment_analyzer'
            )
            
            subreddit = reddit.subreddit(subreddit)
            posts = []
            
            # Search for ticker mentions
            for post in subreddit.search(self.ticker, limit=limit):
                posts.append(post.title)
            
            print(f"Found {len(posts)} Reddit posts")
            return self.analyze_news_headlines(posts)
            
        except Exception as e:
            print(f"Error fetching Reddit posts: {e}")
            return pd.DataFrame()
    
    def analyze_custom_text(self, texts):
        """
        Analyze custom text data
        
        Args:
            texts: List of text strings or DataFrame with 'text' column
        """
        if isinstance(texts, pd.DataFrame):
            texts = texts['text'].tolist()
        
        return self.analyze_news_headlines(texts)
    
    def get_sentiment_summary(self):
        """Get overall sentiment summary"""
        if not self.sentiments:
            return "No sentiment data available"
        
        df = pd.DataFrame(self.sentiments)
        
        summary = {
            'total_analyzed': len(df),
            'avg_textblob_polarity': df['textblob_polarity'].mean(),
            'avg_vader_compound': df['vader_compound'].mean(),
            'textblob_distribution': df['textblob_sentiment'].value_counts().to_dict(),
            'vader_distribution': df['vader_sentiment'].value_counts().to_dict(),
            'positive_ratio': (df['vader_sentiment'] == 'positive').sum() / len(df),
            'negative_ratio': (df['vader_sentiment'] == 'negative').sum() / len(df),
            'neutral_ratio': (df['vader_sentiment'] == 'neutral').sum() / len(df)
        }
        
        return summary
    
    def plot_sentiment_analysis(self):
        """Visualize sentiment analysis results"""
        if not self.sentiments:
            print("No sentiment data to plot!")
            return
        
        df = pd.DataFrame(self.sentiments)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Sentiment Distribution (VADER)
        sentiment_counts = df['vader_sentiment'].value_counts()
        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
        ax1 = axes[0, 0]
        sentiment_counts.plot(kind='bar', ax=ax1, 
                             color=[colors[x] for x in sentiment_counts.index])
        ax1.set_title(f'Sentiment Distribution - {self.ticker}', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=0)
        
        # 2. Score Distribution
        ax2 = axes[0, 1]
        ax2.hist(df['vader_compound'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(df['vader_compound'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["vader_compound"].mean():.3f}')
        ax2.set_title('VADER Compound Score Distribution', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Compound Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Sentiment Pie Chart
        ax3 = axes[1, 0]
        sentiment_counts.plot(kind='pie', ax=ax3, autopct='%1.1f%%',
                             colors=[colors[x] for x in sentiment_counts.index],
                             startangle=90)
        ax3.set_title('Sentiment Percentage', fontweight='bold', fontsize=12)
        ax3.set_ylabel('')
        
        # 4. TextBlob vs VADER comparison
        ax4 = axes[1, 1]
        ax4.scatter(df['textblob_polarity'], df['vader_compound'], 
                   alpha=0.6, c=df['vader_compound'], cmap='RdYlGn', s=50)
        ax4.set_title('TextBlob vs VADER Correlation', fontweight='bold', fontsize=12)
        ax4.set_xlabel('TextBlob Polarity')
        ax4.set_ylabel('VADER Compound')
        ax4.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax4.axvline(0, color='black', linestyle='--', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        summary = self.get_sentiment_summary()
        print("\n" + "="*50)
        print(f"SENTIMENT ANALYSIS SUMMARY - {self.ticker}")
        print("="*50)
        print(f"Total texts analyzed: {summary['total_analyzed']}")
        print(f"\nAverage Scores:")
        print(f"  TextBlob Polarity: {summary['avg_textblob_polarity']:.3f}")
        print(f"  VADER Compound:    {summary['avg_vader_compound']:.3f}")
        print(f"\nSentiment Distribution:")
        print(f"  Positive: {summary['positive_ratio']*100:.1f}%")
        print(f"  Negative: {summary['negative_ratio']*100:.1f}%")
        print(f"  Neutral:  {summary['neutral_ratio']*100:.1f}%")
        
        # Overall sentiment
        if summary['avg_vader_compound'] > 0.05:
            overall = "📈 BULLISH"
        elif summary['avg_vader_compound'] < -0.05:
            overall = "📉 BEARISH"
        else:
            overall = "➡️ NEUTRAL"
        print(f"\nOverall Market Sentiment: {overall}")
        print("="*50)
    
    def generate_wordcloud(self):
        """Generate word cloud from analyzed texts"""
        if not self.sentiments:
            print("No sentiment data available!")
            return
        
        df = pd.DataFrame(self.sentiments)
        all_text = ' '.join(df['text'])
        cleaned_text = self.clean_text(all_text)
        
        wordcloud = WordCloud(width=1200, height=600, 
                             background_color='white',
                             colormap='viridis',
                             max_words=100).generate(cleaned_text)
        
        plt.figure(figsize=(14, 7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud - {self.ticker} Sentiment Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # Print top keywords
        keywords = self.extract_keywords(df['text'].tolist())
        print("\nTop 10 Keywords:")
        for word, count in keywords[:10]:
            print(f"  {word}: {count}")
    
    def compare_with_stock_price(self, period='1mo'):
        """Compare sentiment trend with stock price"""
        if not self.sentiments:
            print("No sentiment data available!")
            return
        
        # Fetch stock data
        stock = yf.Ticker(self.ticker)
        hist = stock.history(period=period)
        
        # Create sentiment timeline
        df_sent = pd.DataFrame(self.sentiments)
        df_sent['date'] = pd.to_datetime(df_sent['timestamp']).dt.date
        daily_sentiment = df_sent.groupby('date')['vader_compound'].mean()
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Stock price
        ax1.plot(hist.index, hist['Close'], linewidth=2, color='blue')
        ax1.set_ylabel('Stock Price ($)', fontweight='bold')
        ax1.set_title(f'{self.ticker} - Stock Price vs Sentiment', 
                     fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Sentiment
        ax2.plot(daily_sentiment.index, daily_sentiment.values, 
                linewidth=2, color='green', marker='o')
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Sentiment Score', fontweight='bold')
        ax2.set_xlabel('Date', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(daily_sentiment.index, daily_sentiment.values, 0, 
                         alpha=0.3, color='green')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filename=None):
        """Export sentiment analysis results to CSV"""
        if not self.sentiments:
            print("No sentiment data to export!")
            return
        
        if filename is None:
            filename = f"{self.ticker}_sentiment_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
        
        df = pd.DataFrame(self.sentiments)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


# Example Usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = MarketSentimentAnalyzer('TSLA')
    
    # Fetch and analyze Yahoo Finance news
    news_df = analyzer.fetch_yahoo_news(max_articles=50)
    
    # You can also analyze custom text
    custom_headlines = [
        "Tesla stock surges on record delivery numbers",
        "Concerns grow over Tesla's declining margins",
        "Musk's latest innovation could revolutionize industry",
        "Tesla faces increased competition in EV market",
        "Analysts upgrade Tesla price target"
    ]
    analyzer.analyze_custom_text(custom_headlines)
    
    # Visualize results
    analyzer.plot_sentiment_analysis()
    
    # Generate word cloud
    analyzer.generate_wordcloud()
    
    # Compare with stock price
    analyzer.compare_with_stock_price(period='1mo')
    
    # Export results
    analyzer.export_results()
    
    # Get detailed results
    results_df = pd.DataFrame(analyzer.sentiments)
    print("\nSample Results:")
    print(results_df.head())