import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from binance.client import Client
import time

# Fungsi untuk mengambil data harga dari Binance
def get_data(symbol, interval='1h', limit=1000, client=None):
    print(f"Downloading market data for {symbol}...")
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    if not klines:
        raise ValueError(f"Data untuk simbol {symbol} tidak ditemukan atau kosong.")
    data = {
        'Date': [pd.to_datetime(k[0], unit='ms') for k in klines],
        'Open': [float(k[1]) for k in klines],
        'High': [float(k[2]) for k in klines],
        'Low': [float(k[3]) for k in klines],
        'Close': [float(k[4]) for k in klines],
        'Volume': [float(k[5]) for k in klines]
    }
    df = pd.DataFrame(data).set_index('Date')
    print(f"Data downloaded: {df.tail()}")
    return df

# Fungsi untuk melatih model HMM
def train_hmm(data, n_components):
    print("Training HMM model...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    model = hmm.GaussianHMM(n_components=n_components)
    model.fit(data_scaled)
    return model, scaler

# Fungsi untuk membuat prediksi
def predict(model, scaler, data):
    print("Predicting hidden states...")
    data_scaled = scaler.transform(data)
    hidden_states = model.predict(data_scaled)
    
    # Probabilitas dari prediksi
    state_probs = model.predict_proba(data_scaled)
    return hidden_states, state_probs

# Fungsi untuk mengambil keputusan trading
def make_decision(hidden_states, state_probs, data):
    # Ambil probabilitas untuk keadaan terakhir
    latest_state_probs = state_probs[-1]
    
    # Contoh logika pengambilan keputusan
    state_labels = ['State 0', 'State 1', 'State 2']  # Ubah sesuai jumlah komponen
    max_prob_state = np.argmax(latest_state_probs)
    action = 'Hold'
    
    if latest_state_probs[max_prob_state] > 0.6:  # Contoh threshold untuk keputusan
        if max_prob_state == 0:
            action = 'Buy'
        elif max_prob_state == 1:
            action = 'Sell'
    
    # Tampilkan persentase probabilitas
    state_probs_summary = dict(zip(state_labels, latest_state_probs))
    print(f"Latest State Probabilities: {state_probs_summary}")
    print(f"Trade decision: {action}")
    
    return action, state_probs_summary

# Fungsi untuk menyesuaikan jumlah agar sesuai dengan LOT_SIZE
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

# Fungsi untuk eksekusi trade order
def execute_trade(action, symbol, quantity, client):
    quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
    print(f"Executing {action} trade for {symbol} with quantity {quantity:.6f}")
    try:
        if action == 'Buy':
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
            print(f"Buy order placed: {order}")
        elif action == 'Sell':
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
            print(f"Sell order placed: {order}")
        else:
            print(f"Unknown action {action}. Skipping trade.")
    except Exception as e:
        print(f"Error executing trade: {e}")

# Fungsi utama
def main():
    # Inisialisasi Binance client
    api_key = 'h6js6UiH8EDXBRhzQYWoYUjBxEisuf0OgD86BD6bcfrn2UAvx7sYBShd8LIoOj2a'
    api_secret = 'Sg6yoywPejPggWekj40oGHz1vQivrg5tNoSXyWVFcsqPgUmcxCEbUjvI1KyOg1TS'
    client = Client(api_key, api_secret)

    # Definisikan simbol trading
    symbol = 'DOGEUSDT'

    # Latih model HMM dengan data historis
    historical_data = get_data(symbol, '1h', 1000, client)  # Ambil data historis untuk melatih model
    model, scaler = train_hmm(historical_data[['Close']], n_components=3)

    while True:
        try:
            # Ambil data harga terbaru
            new_data = get_data(symbol, '1h', 1000, client)  # Ambil data hingga waktu terkini

            # Latih ulang model dengan data terbaru (opsional)
            model, scaler = train_hmm(new_data[['Close']], n_components=3)

            # Buat prediksi
            hidden_states, state_probs = predict(model, scaler, new_data[['Close']])
            
            # Ambil keputusan trading
            action, state_probs_summary = make_decision(hidden_states, state_probs, new_data)
            
            # Tentukan jumlah untuk trading (contoh jumlah)
            quantity = 1.0

            # Eksekusi trade
            execute_trade(action, symbol, quantity, client)

        except Exception as e:
            print(f"Error in main loop: {e}")

        # Tunggu sebelum iterasi berikutnya (misalnya, 15 menit)
        time.sleep(15 * 60)

if __name__ == "__main__":
    main()
