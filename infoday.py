import requests
import time
import numpy as np
import pickle
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from datetime import datetime, timedelta
import nltk

# Pastikan VADER lexicon sudah diunduh
nltk.download('vader_lexicon')

# API Key untuk NewsAPI
api_key = '8c829b1bdcfe4f12ada6688a781e12cc'

def get_news(queries, language='en', from_date=None, page_size=10):
    articles = []
    if not from_date:
        from_date = datetime.now().strftime('%Y-%m-%d')
    
    for query in queries:
        url = f'https://newsapi.org/v2/everything?q={query}&from={from_date}&to={from_date}&language={language}&pageSize={page_size}&apiKey={api_key}'
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
    return any(keyword.lower() in title or keyword.lower() in description for keyword in keywords) or \
           any(term.lower() in title or term.lower() in description for term in price_related_terms)

def analyze_sentiment_vader(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

def extract_features(articles, vectorizer=None):
    texts = [article['title'] + ' ' + article['description'] for article in articles]
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
    else:
        X = vectorizer.transform(texts)
    return X, vectorizer

def train_model(X, y):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)
    return clf

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

def make_trading_decision_with_model(model, X_test):
    predictions_proba = model.predict_proba(X_test)
    predicted_class = model.predict(X_test)
    
    # Ambil probabilitas untuk kelas yang diprediksi
    max_proba = np.max(predictions_proba)
    confidence_percentage = max_proba * 100

    if predicted_class[0] == 1:
        action = "Buy"
    elif predicted_class[0] == -1:
        action = "Sell"
    else:
        action = "Hold"

    return action, confidence_percentage

keywords = ["dogecoin", "dogeusdt", "cryptocurrency", "price", "market", "value", "investment", "news"]
price_related_terms = ["price", "market", "value", "investment"]

# Load atau train model
if os.path.exists('model.pkl'):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
else:
    # Persiapkan data pelatihan (X_train, y_train) untuk contoh
    dummy_articles = [
        {'title': 'Dogecoin rises in price', 'description': 'The price of Dogecoin has increased significantly.'},
        {'title': 'Market trends for cryptocurrency', 'description': 'The market trends show a rise in value.'},
        {'title': 'Investment in Dogecoin', 'description': 'Investors are showing interest in Dogecoin.'},
        {'title': 'Cryptocurrency market crash', 'description': 'The market has experienced a significant drop in value.'},
        {'title': 'Dogecoin stability', 'description': 'Dogecoin shows stable performance despite market fluctuations.'}
    ]
    
    dummy_labels = [1, 1, 1, -1, 0]  # 1 untuk Buy, -1 untuk Sell, 0 untuk Hold
    
    # Ekstrak fitur dan latih model
    X_train, vectorizer = extract_features(dummy_articles)
    y_train = np.array(dummy_labels)
    
    model = train_model(X_train, y_train
