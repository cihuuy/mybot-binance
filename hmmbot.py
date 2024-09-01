import pandas as pd
import numpy as np
import talib
import pickle
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from hmmlearn import hmm
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime

# Fungsi untuk mendapatkan data pasar dari Binance API
def get_market_data(symbol, client, interval='15m', limit=1000):
    print(f"Mengunduh data pasar untuk {symbol} menggunakan Binance API...")
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    if not klines:
        raise ValueError(f"Data untuk simbol {symbol} tidak ditemukan atau kosong.")
    data = {
        'Date': [datetime.fromtimestamp(k[0] / 1000.0) for k in klines],
        'Open': [float(k[1]) for k in klines],
        'High': [float(k[2]) for k in klines],
        'Low': [float(k[3]) for k in klines],
        'Close': [float(k[4]) for k in klines],
        'Volume': [float(k[5]) for k in klines]
    }
    df = pd.DataFrame(data).set_index('Date')
    print(f"Data yang diunduh: {df.head()}")
    return df

# Fungsi untuk menambahkan indikator teknis ke data pasar
def add_technical_indicators(data):
    print("Menambahkan indikator teknis...")
    data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_SIGNAL'], data['MACD_HIST'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    return data

# Fungsi untuk melatih model AI
def train_ai_model(data):
    print("Melatih model AI...")
    data = data.dropna()
    X = data[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']]
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_distributions = {
        'n_estimators': [100, 200, 500, 1000],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        GradientBoostingClassifier(),
        param_distributions=param_distributions,
        n_iter=100,
        cv=tscv,
        scoring='accuracy',
        verbose=2,
        random_state=42
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    model_accuracy = accuracy_score(y_test, best_model.predict(X_test))
    print(f"Akurasi model: {model_accuracy * 100:.2f}%")
    print("Parameter terbaik yang ditemukan: ", random_search.best_params_)
    print("Skor terbaik: ", random_search.best_score_)

    # Simpan model dan scaler
    with open('gradient_boosting_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)
    with open('gradient_boosting_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    return best_model, scaler, model_accuracy

# Fungsi untuk backtest model
def backtest_model(data, model, scaler):
    print("Melakukan backtest model...")
    data = data.dropna()
    X = data[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']]
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    backtest_accuracy = accuracy_score(y, y_pred)
    print(f"Akurasi backtest: {backtest_accuracy * 100:.2f}%")
    return backtest_accuracy

# Fungsi untuk memuat model dan scaler yang disimpan
def load_model_and_scaler():
    try:
        with open('gradient_boosting_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('gradient_boosting_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except FileNotFoundError:
        print("File model atau scaler tidak ditemukan. Melatih model baru.")
        return None, None

# Fungsi untuk melatih model HMM
def train_hmm(data, n_components):
    print("Melatih model HMM...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    model = hmm.GaussianHMM(n_components=n_components)
    model.fit(data_scaled)
    return model, scaler

# Fungsi untuk melakukan prediksi dengan model HMM
def predict_hmm(model, scaler, data):
    print("Memprediksi keadaan tersembunyi dengan HMM...")
    data_scaled = scaler.transform(data)
    hidden_states = model.predict(data_scaled)
    
    # Probabilitas keadaan tersembunyi
    state_probs = model.predict_proba(data_scaled)
    return hidden_states, state_probs

# Fungsi untuk membuat keputusan perdagangan berdasarkan perbandingan model
def make_trade_decision(data, gb_model, gb_scaler, hmm_model, hmm_scaler):
    # Keputusan menggunakan AI (Gradient Boosting)
    X = data[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']].dropna()
    X_scaled = gb_scaler.transform(X)
    gb_prediction = gb_model.predict(X_scaled[-1:])
    gb_action = 'Buy' if gb_prediction == 1 else 'Sell'
    
    # Keputusan menggunakan HMM
    hmm_hidden_states, hmm_state_probs = predict_hmm(hmm_model, hmm_scaler, data[['Close']])
    latest_hmm_state_probs = hmm_state_probs[-1]
    hmm_action = 'Hold'
    
    if latest_hmm_state_probs[0] > 0.6:
        hmm_action = 'Buy'
    elif latest_hmm_state_probs[1] > 0.6:
        hmm_action = 'Sell'
    
    print(f"Keputusan Gradient Boosting: {gb_action}")
    print(f"Keputusan HMM: {hmm_action}")
    
    # Perbandingan keputusan
    if gb_action == 'Buy' and hmm_action == 'Buy':
        final_decision = 'Buy'
    elif gb_action == 'Buy' and hmm_action == 'Sell':
        final_decision = 'Hold'
    elif gb_action == 'Sell' and hmm_action == 'Buy':
        final_decision = 'Hold'
    elif gb_action == 'Sell' and hmm_action == 'Sell':
        final_decision = 'Sell'
    elif gb_action == 'Buy' and hmm_action == 'Hold':
        final_decision = 'Hold' 
    elif gb_action == 'Sell' and hmm_action == 'Hold':
        final_decision = 'Hold' 
    else:  # gb_action == 'Sell' and hmm_action == 'Sell'
        final_decision = 'Hold'
    
    print(f"Keputusan akhir perdagangan: {final_decision}")
    
    # Menghitung stop_loss dan take_profit
    stop_loss_pct = gb_model.predict_proba(X_scaled[-1:])[0][1] * 0.02
    take_profit_pct = gb_model.predict_proba(X_scaled[-1:])[0][0] * 0.02
    current_price = data['Close'].iloc[-1]
    stop_loss = current_price * (1 - stop_loss_pct)
    take_profit = current_price * (1 + take_profit_pct)
    
    return final_decision, stop_loss, take_profit

# Fungsi untuk menyesuaikan kuantitas agar sesuai dengan LOT_SIZE
def adjust_quantity_to_lot_size(symbol, quantity, client):
    exchange_info = client.get_symbol_info(symbol)
    lot_size_info = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', exchange_info['filters']), None)
    if lot_size_info:
        step_size = float(lot_size_info['stepSize'])
        quantity = round(quantity - (quantity % step_size), len(str(step_size).split('.')[1]))
    return quantity

# Fungsi untuk memeriksa saldo
def check_balance(asset, client):
    balance_info = client.get_asset_balance(asset=asset)
    balance = float(balance_info['free'])
    return balance

# Fungsi utama trading bot
def trade_bot(symbol, client):
    # Muat atau latih model AI dan scaler
    gb_model, gb_scaler = load_model_and_scaler()
    if gb_model is None or gb_scaler is None:
        df = get_market_data(symbol, client)
        df = add_technical_indicators(df)
        gb_model, gb_scaler, _ = train_ai_model(df)
    
    # Dapatkan data pasar terbaru
    df = get_market_data(symbol, client)
    df = add_technical_indicators(df)
    
    # Melatih model HMM
    hmm_model, hmm_scaler = train_hmm(df[['Close']], n_components=2)
    
    # Buat keputusan perdagangan berdasarkan perbandingan model
    decision, stop_loss, take_profit = make_trade_decision(df, gb_model, gb_scaler, hmm_model, hmm_scaler)
    
    if decision == 'Buy':
        # Memastikan hanya membeli dengan saldo yang tersedia
        base_asset = symbol[:-4]
        quote_asset = symbol[-4:]
        balance = check_balance(quote_asset, client)
        if balance <= 0:
            print(f"Tidak ada saldo yang tersedia untuk membeli {base_asset}.")
            return
        
        price = df['Close'].iloc[-1]
        quantity = balance / price
        quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
        min_notional = float(client.get_symbol_info(symbol)['filters'][2]['minNotional'])
        if quantity * price < min_notional:
            print(f"Jumlah minimum untuk diperdagangkan adalah {min_notional}.")
            return
        
        # Pasang order beli
        try:
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
            print(f"Order beli berhasil: {order}")
        except BinanceAPIException as e:
            print(f"Error saat melakukan order beli: {str(e)}")
            return
        
    elif decision == 'Sell':
        # Memastikan hanya menjual dengan saldo yang tersedia
        balance = check_balance(symbol[:-4], client)
        if balance <= 0:
            print(f"Tidak ada saldo yang tersedia untuk menjual {symbol[:-4]}.")
            return
        
        quantity = adjust_quantity_to_lot_size(symbol, balance, client)
        
        # Pasang order jual
        try:
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
            print(f"Order jual berhasil: {order}")
        except BinanceAPIException as e:
            print(f"Error saat melakukan order jual: {str(e)}")
            return

    print(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")

# Inisialisasi API Binance
api_key = 'h6js6UiH8EDXBRhzQYWoYUjBxEisuf0OgD86BD6bcfrn2UAvx7sYBShd8LIoOj2a'
api_secret = 'Sg6yoywPejPggWekj40oGHz1vQivrg5tNoSXyWVFcsqPgUmcxCEbUjvI1KyOg1TS'
client = Client(api_key, api_secret)

# Jalankan bot trading setiap 15 menit
while True:
    try:
        server_time = client.get_server_time()
        print(f"Waktu server Binance: {datetime.fromtimestamp(server_time['serverTime'] / 1000)}")
        trade_bot('DOGEUSDT', client)
    except Exception as e:
        print(f"Error: {str(e)}")
    time.sleep(900)  # 15 menit
