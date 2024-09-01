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
        # Mengatur tanggal satu hari yang lalu
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

def is_relevant_article(article, keywords, price_related_terms):
    title = article.get('title', '').lower()
    description = article.get('description', '').lower()
    
    # Memeriksa apakah artikel relevan berdasarkan kata kunci dan istilah terkait harga
    return any(keyword.lower() in title or keyword.lower() in description for keyword in keywords) or \
           any(term.lower() in title or term.lower() in description for term in price_related_terms)

# Analisis Sentimen dengan VADER
def analyze_sentiment_vader(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# Membuat keputusan trading berdasarkan sentimen pasar
def make_trading_decision(sentiments):
    average_sentiment = sum([s['compound'] for s in sentiments]) / len(sentiments) if sentiments else 0
    percentage = (average_sentiment + 1) * 50  # Mengubah nilai -1 to 1 menjadi 0 to 100 persen
    
    if average_sentiment > 0.05:
        action = "Buy"
    elif average_sentiment < -0.05:
        action = "Sell"
    else:
        action = "Hold"
    
    return action, percentage, average_sentiment

# Kata kunci yang ingin dicari
keywords = ["dogecoin", "dogeusdt", "cryptocurrency", "price", "market", "value", "investment", "news"]
price_related_terms = ["price", "market", "value", "investment"]

while True:
    # Mendapatkan berita dari satu hari terakhir
    articles = get_news(keywords, page_size=5)

    if articles:
        relevant_articles = [article for article in articles if is_relevant_article(article, keywords, price_related_terms)]
        
        if relevant_articles:
            sentiments = []
            for article in relevant_articles:
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
            decision, percentage, total_compound = make_trading_decision(sentiments)
            print(f"Trading Decision: {decision}")
            print(f"Confidence Percentage: {percentage:.2f}%")
            print(f"Total Compound Sentiment: {total_compound:.4f}")

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
            print("No relevant articles found.")
    else:
        print("Tidak ada artikel ditemukan.")

    # Tunggu 1 jam sebelum loop berikutnya
    time.sleep(3600)
