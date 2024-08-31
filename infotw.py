import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Pastikan untuk mengunduh dataset VADER terlebih dahulu
nltk.download('vader_lexicon')

# Konfigurasi API Twitter (ganti dengan API key Anda sendiri)
consumer_key = 'lZn1GEqhvdBUnE8svfWn8O6ua'
consumer_secret = 'RMbVP8yQ2flcTTM0ebuWRGQ9b993yinZjTTRIqWm8SFnys5bJC'
access_token = '1786582804786200576-VNzTlYH3gOFZC7MH2Len6Lurfwmp9f'
access_token_secret = '1Qh3KRSEvydtAE5oqbk7Dztu02WNmrWtkYA1uJcgGi3xw'

# Autentikasi Twitter API
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Fungsi untuk mengambil tweet terkait aset kripto
def get_tweets(keyword, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword, lang="en").items(count)
    tweet_list = [tweet.text for tweet in tweets]
    return tweet_list

# Analisis sentimen menggunakan VADER
def analyze_sentiment_vader(tweet):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(tweet)
    return sentiment

# Analisis sentimen menggunakan TextBlob
def analyze_sentiment_textblob(tweet):
    analysis = TextBlob(tweet)
    sentiment = {
        'polarity': analysis.polarity,
        'subjectivity': analysis.subjectivity
    }
    return sentiment

# Contoh penggunaan
if __name__ == "__main__":
    keyword = 'Dogecoin'  # Ganti dengan aset kripto yang Anda inginkan
    tweets = get_tweets(keyword, count=100)

    for tweet in tweets:
        vader_sentiment = analyze_sentiment_vader(tweet)
        textblob_sentiment = analyze_sentiment_textblob(tweet)
        
        print(f"Tweet: {tweet}")
        print(f"VADER Sentiment: {vader_sentiment}")
        print(f"TextBlob Sentiment: {textblob_sentiment}")
        print("-" * 50)
