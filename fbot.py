from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from hmmlearn import hmm
from binance.client import Client
from binance.exceptions import BinanceAPIException
from datetime import datetime
import pandas as pd
import numpy as np
import talib
import pickle
import time

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
        final_decision = 'Sell'
        stop_loss_enabled = True
        take_profit_enabled = True
    elif gb_action == 'Sell' and hmm_action == 'Sell':
        final_decision = 'Sell'
        stop_loss_enabled = False
        take_profit_enabled = False
    elif gb_action == 'Buy' and hmm_action == 'Hold':
        final_decision = 'Buy'
    elif gb_action == 'Sell' and hmm_action == 'Hold':
        final_decision = 'Sell'
        stop_loss_enabled = True
        take_profit_enabled = True
    else:
        final_decision = 'Hold'
    
    print(f"Keputusan akhir perdagangan: {final_decision}")
    
    # Menghitung stop_loss dan take_profit
    stop_loss_pct = gb_model.predict_proba(X_scaled[-1:])[0][1] * 0.02
    take_profit_pct = gb_model.predict_proba(X_scaled[-1:])[0][0] * 0.02
    current_price = data['Close'].iloc[-1]
    stop_loss = current_price * (1 - stop_loss_pct) if 'stop_loss_enabled' in locals() and stop_loss_enabled else None
    take_profit = current_price * (1 + take_profit_pct) if 'take_profit_enabled' in locals() and take_profit_enabled else None
    
    print(f"Stop Loss: {stop_loss}, Take Profit: {take_profit}")
    
    return final_decision, stop_loss, take_profit

# Fungsi untuk mengeksekusi perdagangan berdasarkan keputusan
def execute_trade(decision, stop_loss, take_profit, symbol, client):
    print(f"Mengeksekusi perdagangan: {decision} untuk {symbol}")
    
    try:
        symbol_info = client.get_symbol_info(symbol)
        lot_size_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
        min_qty = float(lot_size_filter['minQty'])
        step_size = float(lot_size_filter['stepSize'])
        
        if decision == 'Buy':
            # Dapatkan balance yang tersedia
            balance = client.get_asset_balance(asset='USDT')
            available_balance = float(balance['free'])
            
            # Hitung jumlah yang dapat dibeli berdasarkan balance yang tersedia
            last_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            quantity = available_balance / last_price
            
            # Adjust the quantity to be within LOT_SIZE constraints
            quantity = max(min_qty, round(quantity / step_size) * step_size)
            
            if quantity >= min_qty:
                print(f"Menempatkan order beli untuk {symbol} dengan jumlah {quantity}...")
                order = client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity
                )
                print("Order berhasil ditempatkan:", order)
            else:
                print("Tidak cukup balance untuk melakukan pembelian.")
                
        elif decision == 'Sell':
            # Dapatkan balance koin yang akan dijual
            balance = client.get_asset_balance(asset=symbol.replace('USDT', ''))
            available_balance = float(balance['free'])
            
            # Adjust the quantity to be within LOT_SIZE constraints
            quantity = max(min_qty, round(available_balance / step_size) * step_size)
            
            if quantity >= min_qty:
                print(f"Menempatkan order jual untuk {symbol} dengan jumlah {quantity}...")
                order = client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
                print("Order berhasil ditempatkan:", order)
            else:
                print("Tidak ada koin yang dapat dijual.")
                
        else:
            print("Keputusan adalah Hold. Tidak ada tindakan yang dilakukan.")
            
    except BinanceAPIException as e:
        print(f"Binance API Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


# Main trading loop
def main_trading_loop(symbol, client, interval=900):
    while True:
        try:
            # Mendapatkan data pasar
            market_data = get_market_data(symbol, client)

            # Menambahkan indikator teknis
            market_data_with_indicators = add_technical_indicators(market_data)

            # Memuat atau melatih model AI
            gb_model, gb_scaler = load_model_and_scaler()
            if gb_model is None or gb_scaler is None:
                gb_model, gb_scaler, _ = train_ai_model(market_data_with_indicators)

            # Backtest model
            backtest_accuracy = backtest_model(market_data_with_indicators, gb_model, gb_scaler)

            # Melatih model HMM
            hmm_model, hmm_scaler = train_hmm(market_data[['Close']], n_components=2)

            # Membuat keputusan perdagangan
            final_decision, stop_loss, take_profit = make_trade_decision(market_data_with_indicators, gb_model, gb_scaler, hmm_model, hmm_scaler)

            # Mengeksekusi perdagangan berdasarkan keputusan
            execute_trade(final_decision, stop_loss, take_profit, symbol, client)

        except Exception as e:
            print(f"Error during trading loop: {e}")
        
        # Tunggu selama interval sebelum melakukan perdagangan berikutnya
        time.sleep(interval)

# Pengaturan dan inisialisasi
api_key = 'h6js6UiH8EDXBRhzQYWoYUjBxEisuf0OgD86BD6bcfrn2UAvx7sYBShd8LIoOj2a'
api_secret = 'Sg6yoywPejPggWekj40oGHz1vQivrg5tNoSXyWVFcsqPgUmcxCEbUjvI1KyOg1TS'
symbol = 'DOGEUSDT'
client = Client(api_key, api_secret)

# Menjalankan trading loop
main_trading_loop(symbol, client)
