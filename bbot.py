import yfinance as yf
import talib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
import pickle
from datetime import datetime, timedelta

# Mendapatkan waktu pasar dari Binance API
def get_market_time(client):
    server_time = client.get_server_time()
    market_time = datetime.fromtimestamp(server_time['serverTime'] / 1000.0)
    return market_time

# Mendapatkan data pasar dari Binance API
def get_market_data(symbol, client, interval='1h', limit=1000):
    print(f"Downloading market data for {symbol} using Binance API...")
    
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    
    if not klines:
        raise ValueError(f"Data untuk simbol {symbol} tidak ditemukan atau kosong.")
    
    data = {
        'Open': [],
        'High': [],
        'Low': [],
        'Close': [],
        'Volume': [],
        'Date': []
    }
    
    for kline in klines:
        data['Date'].append(datetime.fromtimestamp(kline[0] / 1000.0))
        data['Open'].append(float(kline[1]))
        data['High'].append(float(kline[2]))
        data['Low'].append(float(kline[3]))
        data['Close'].append(float(kline[4]))
        data['Volume'].append(float(kline[5]))
    
    data = pd.DataFrame(data)
    data.set_index('Date', inplace=True)
    print(f"Data downloaded: {data.head()}")
    
    return data

# Menambahkan indikator teknikal
def add_technical_indicators(data):
    print("Adding technical indicators...")
    data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_SIGNAL'], data['MACD_HIST'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    return data

# Melatih model AI dengan GridSearchCV dan TimeSeriesSplit
def train_ai_model(data):
    print("Training AI model...")
    data = data.dropna()
    if data.empty:
        raise ValueError("Data setelah penambahan indikator teknikal kosong.")
    
    X = data[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']]
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    if X.empty or y.empty:
        raise ValueError("Data fitur atau label kosong.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=tscv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    model = grid_search.best_estimator_
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # Simpan model dan scaler
    with open('trading_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    
    return model, scaler, X_test, y_test

# Memuat model dan scaler yang disimpan
def load_model_and_scaler():
    try:
        with open('trading_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except FileNotFoundError:
        return None, None

# Mengecek saldo akun sebelum melakukan trading dan menyesuaikan kuantitas
def check_balance(symbol, quantity, action, client):
    asset = symbol.replace("USDT", "")
    print(f"Checking balance for {asset}...")
    balances = client.get_asset_balance(asset=asset)
    if balances is None:
        raise ValueError(f"Tidak dapat mengambil saldo untuk {asset}.")
    
    available_balance = float(balances['free'])
    
    if action == 'Sell':
        # Pastikan saldo memenuhi batas minimum lot size untuk penjualan
        lot_size_info = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', client.get_symbol_info(symbol)['filters']), None)
        if lot_size_info is not None:
            min_qty = float(lot_size_info['minQty'])
            if available_balance < min_qty:
                print(f"Saldo {asset} tidak mencukupi untuk menjual. Saldo saat ini: {available_balance} {asset}")
                return None  # Menghindari eksekusi order jual
        if available_balance < quantity:
            print(f"Saldo tidak mencukupi untuk menjual {quantity} {asset}. Menyesuaikan jumlah menjadi {available_balance} {asset}.")
            quantity = available_balance
    
    elif action == 'Buy':
        required_balance = quantity * float(client.get_symbol_ticker(symbol=symbol)['price'])
        if available_balance < required_balance:
            print(f"Saldo tidak mencukupi untuk membeli {quantity} {asset}. Saldo saat ini: {available_balance} {asset}")
            return None  # Menghindari eksekusi order beli
    
    return quantity

# Menyesuaikan kuantitas agar sesuai dengan LOT_SIZE
def adjust_quantity_to_lot_size(symbol, quantity, client):
    exchange_info = client.get_symbol_info(symbol)
    lot_size_info = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', exchange_info['filters']), None)
    
    if lot_size_info is not None:
        min_qty = float(lot_size_info['minQty'])
        max_qty = float(lot_size_info['maxQty'])
        step_size = float(lot_size_info['stepSize'])
        
        if quantity < min_qty:
            quantity = min_qty
        elif quantity > max_qty:
            quantity = max_qty
        else:
            quantity = (quantity // step_size) * step_size
        
        quantity = round(quantity, len(str(step_size).split('.')[1]))
    
    return quantity

# Membuat keputusan trading
def make_trade_decision(data, model, scaler):
    print("Making trade decision...")
    X = data[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']].dropna()
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
    print(f"Executing {action} trade...")
    if action == 'Buy':
        quantity = check_balance('USDT', quantity, action, client)
        if quantity is None:
            print("Tidak cukup saldo USDT untuk membeli.")
            return
        quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
        try:
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
            print(f"Executed Buy order: {order}")
        except BinanceAPIException as e:
            if e.code == -1013 and 'NOTIONAL' in e.message:
                print(f"Error saat eksekusi trade: {e}. Saldo tidak mencukupi untuk melakukan buy.")
            else:
                print(f"Error saat eksekusi trade: {e}")
            order = None
    elif action == 'Sell':
        quantity = check_balance(symbol, quantity, action, client)
        if quantity is None:
            print("Tidak cukup saldo DOGE untuk menjual.")
            return
        quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
        try:
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
            print(f"Executed Sell order: {order}")
        except BinanceAPIException as e:
            if e.code == -1013 and 'LOT_SIZE' in e.message:
                print(f"Error saat eksekusi trade: {e}. Saldo tidak mencukupi untuk menjual.")
            else:
                print(f"Error saat eksekusi trade: {e}")
            order = None
    else:
        print(f"Unknown action {action}")
        order = None
    
    return order

# Fungsi utama untuk menjalankan proses trading
def main(api_key, api_secret, symbol):
    client = Client(api_key, api_secret)
    
    while True:
        market_time = get_market_time(client)
        print(f"Market time: {market_time}")
        
        # Unduh data pasar dan latih model
        data = get_market_data(symbol, client)
        data = add_technical_indicators(data)
        
        model, scaler = load_model_and_scaler()
        if model is None or scaler is None:
            model, scaler, X_test, y_test = train_ai_model(data)
        
        # Buat keputusan trading
        action = make_trade_decision(data, model, scaler)
        
        # Tentukan kuantitas perdagangan
        quantity = 1  # Sesuaikan dengan logika atau strategi perdagangan Anda
        
        if model is not None and scaler is not None:
            if accuracy_score(y_test, model.predict(scaler.transform(X_test))) < 0.51:
                print("Akurasi model atau backtest kurang dari 51%. Melewati perdagangan.")
            else:
                execute_trade(action, symbol, quantity, client)
        
        # Tunggu 1 jam sebelum iterasi berikutnya
        time.sleep(3600)

# Jalankan fungsi utama
api_key = 'h6js6UiH8EDXBRhzQYWoYUjBxEisuf0OgD86BD6bcfrn2UAvx7sYBShd8LIoOj2a'
api_secret = 'Sg6yoywPejPggWekj40oGHz1vQivrg5tNoSXyWVFcsqPgUmcxCEbUjvI1KyOg1TS'
symbol = 'DOGEUSDT'
main(api_key, api_secret, symbol)
