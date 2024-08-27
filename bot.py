import yfinance as yf
import talib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from binance.client import Client

# Mendapatkan data pasar
def get_market_data(symbol, period='1mo', interval='1h'):
    data = yf.download(symbol, period=period, interval=interval)
    return data

# Menambahkan indikator teknikal
def add_technical_indicators(data):
    data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    return data

# Melatih model AI
def train_ai_model(data):
    data = data.dropna()
    X = data[['SMA', 'RSI']]
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    return model, scaler

# Membuat keputusan trading
def make_trade_decision(data, model, scaler):
    X = data[['SMA', 'RSI']].dropna()
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled[-1:])
    
    if prediction == 1:
        action = 'Buy'
    else:
        action = 'Sell'
    
    print(f"Trade decision: {action}")
    return action

# Eksekusi order trading
def execute_trade(action, symbol, quantity, client):
    if action == 'Buy':
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
    elif action == 'Sell':
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
    
    print(f"Executed {action} order: {order}")
    return order

# Menjalankan bot trading
def run_trading_bot(symbol, quantity, client):
    data = get_market_data(symbol)
    data = add_technical_indicators(data)
    
    model, scaler = train_ai_model(data)
    
    action = make_trade_decision(data, model, scaler)
    
    execute_trade(action, symbol, quantity, client)

# Sampel eksekusi
if __name__ == "__main__":
    # Ganti 'api_key' dan 'api_secret' dengan kunci API Binance Anda
    client = Client('api_key', 'api_secret')

    # Menjalankan bot untuk pasangan BTC-USD dengan kuantitas 0.001
    run_trading_bot('BTC-USD', 0.001, client)
