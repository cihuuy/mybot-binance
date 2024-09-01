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

# Fungsi untuk membuat keputusan perdagangan
def make_trade_decision(data, model, scaler, use_hmm=False):
    if use_hmm:
        # Keputusan menggunakan HMM
        hidden_states, state_probs = predict_hmm(model, scaler, data[['Close']])
        latest_state_probs = state_probs[-1]
        
        state_labels = ['State 0', 'State 1', 'State 2']
        max_prob_state = np.argmax(latest_state_probs)
        action = 'Hold'
        
        if latest_state_probs[max_prob_state] > 0.6:
            if max_prob_state == 0:
                action = 'Buy'
            elif max_prob_state == 1:
                action = 'Sell'
        
        state_probs_summary = dict(zip(state_labels, latest_state_probs))
        print(f"Probabilitas Keadaan Terbaru: {state_probs_summary}")
        print(f"Keputusan perdagangan (HMM): {action}")
        return action, state_probs_summary
    
    else:
        # Keputusan menggunakan AI
        X = data[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']].dropna()
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled[-1:])
        stop_loss_pct = model.predict_proba(X_scaled[-1:])[0][1] * 0.02  # Contoh multiplier untuk penyesuaian risiko
        take_profit_pct = model.predict_proba(X_scaled[-1:])[0][0] * 0.02  # Contoh multiplier untuk penyesuaian reward
        current_price = data['Close'].iloc[-1]
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)
        action = 'Buy' if prediction == 1 else 'Sell'
        print(f"Keputusan perdagangan (AI): {action}")
        print(f"Stop-Loss yang dihitung: {stop_loss:.6f}")
        print(f"Take-Profit yang dihitung: {take_profit:.6f}")
        return action, stop_loss, take_profit

# Fungsi untuk menyesuaikan kuantitas agar sesuai dengan LOT_SIZE
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
        quantity = round(quantity - (quantity % step_size), 8)
    return quantity

# Fungsi untuk menghitung jumlah kuantitas pembelian
def calculate_quantity(symbol, price, client):
    balance_info = client.get_asset_balance(asset='USDT')
    available_balance = float(balance_info['free'])
    quantity = available_balance / price
    quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
    return quantity

# Fungsi untuk menempatkan order dengan kuantitas yang disesuaikan
def place_order(client, symbol, action, price, stop_loss, take_profit, quantity):
    try:
        if action == 'Buy':
            # Membuat order beli pasar dengan kuantitas yang disesuaikan
            order = client.order_market_buy(
                symbol=symbol,
                quantity=quantity
            )
        elif action == 'Sell':
            # Membuat order jual pasar dengan kuantitas yang disesuaikan
            order = client.order_market_sell(
                symbol=symbol,
                quantity=quantity
            )
        else:
            print("Tidak ada aksi yang valid ditentukan.")
            return None

        print(f"Order {action} berhasil ditempatkan dengan ID order: {order['orderId']}")
        print(f"Stop-Loss: {stop_loss}")
        print(f"Take-Profit: {take_profit}")

    except BinanceAPIException as e:
        print(f"Terjadi kesalahan saat menempatkan order {action}: {str(e)}")

def main():
    api_key = "your_binance_api_key"
    api_secret = "your_binance_api_secret"
    client = Client(api_key, api_secret)

    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        symbol = 'DOGEUSDT'
        data = get_market_data(symbol, client)
        data = add_technical_indicators(data)
        model, scaler, _ = train_ai_model(data)

    while True:
        try:
            symbol = 'DOGEUSDT'
            data = get_market_data(symbol, client)
            data = add_technical_indicators(data)

            action, stop_loss, take_profit = make_trade_decision(data, model, scaler)

            price = data['Close'].iloc[-1]
            quantity = calculate_quantity(symbol, price, client)

            if quantity > 0:
                place_order(client, symbol, action, price, stop_loss, take_profit, quantity)

            print("Menunggu 15 menit sebelum eksekusi berikutnya.")
            time.sleep(15 * 60)  # Tunggu 15 menit

        except Exception as e:
            print(f"Terjadi kesalahan: {str(e)}")
            time.sleep(60)  # Tunggu 1 menit sebelum mencoba lagi jika terjadi kesalahan

if __name__ == "__main__":
    main()
