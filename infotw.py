import requests
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta

# Pastikan VADER lexicon sudah diunduh
nltk.download('vader_lexicon')

# API Key untuk NewsAPI
api_key = '8c829b1bdcfe4f12ada6688a781e12cc'

def get_news(queries, language='en', from_date=None, page_size=10):
    articles = []
    if not from_date:
        # Mengatur tanggal satu minggu yang lalu
        from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    for query in queries:
        url = f'https://newsapi.org/v2/everything?q={query}&from={from_date}&language={language}&pageSize={page_size}&apiKey={api_key}'
        response = requests.get(url)
        
        if response.status_code == 200:
            news_data = response.json()
            articles.extend(news_data['articles'])
        else:
            print(f"Error fetching data for {query}: {response.status_code}")
    return articles

# Analisis Sentimen dengan VADER
def analyze_sentiment_vader(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# Membuat keputusan trading berdasarkan sentimen pasar
def make_trading_decision(sentiments):
    average_sentiment = sum([s['compound'] for s in sentiments]) / len(sentiments)
    percentage = (average_sentiment + 1) * 50  # Mengubah nilai -1 to 1 menjadi 0 to 100 persen
    
    if average_sentiment > 0.05:
        action = "Buy"
    elif average_sentiment < -0.05:
        action = "Sell"
    else:
        action = "Hold"
    
    return action, percentage

# Kata kunci yang ingin dicari
keywords = ["Dogecoin", "DOGEUSDT", "cryptocurrency"]

while True:
    # Mendapatkan berita dari satu minggu terakhir
    articles = get_news(keywords, page_size=5)

    if articles:
        sentiments = []
        for article in articles:
            title = article['title']
            description = article['description'] if article['description'] else ""
            source = article['source']['name']
            published_at = article['publishedAt']
            content = title + ". " + description
            sentiment = analyze_sentiment_vader(content)
            sentiments.append(sentiment)

            print(f"Title: {title}")
            print(f"Description: {description}")
            print(f"Source: {source}")
            print(f"Published At: {published_at}")
            print(f"Sentiment (VADER): {sentiment}")
            print("---")

        # Membuat keputusan trading
        decision, percentage = make_trading_decision(sentiments)
        print(f"Trading Decision: {decision}")
        print(f"Confidence Percentage: {percentage:.2f}%")

        # Implementasikan eksekusi trading berdasarkan keputusan
        # Misalnya, eksekusi order di Binance API
        
        if decision == "Buy":
            # Panggil fungsi untuk membeli DOGE
            print("Executing Buy trade...")
            # buy_doge()
        elif decision == "Sell":
            # Panggil fungsi untuk menjual DOGE
            print("Executing Sell trade...")
            # sell_doge()
        else:
            print("Holding position. No trade executed.")
        
    else:
        print("Tidak ada artikel ditemukan.")

    # Tunggu 1 jam sebelum loop berikutnya
    time.sleep(3600)
