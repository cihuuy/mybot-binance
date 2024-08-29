import yfinance as yf
import talib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time

# Mendapatkan data pasar dari yfinance
def get_market_data(symbol, period='1mo', interval='1h'):
    data = yf.download(symbol, period=period, interval=interval)
    if data.empty:
        raise ValueError(f"Data untuk simbol {symbol} tidak ditemukan atau kosong.")
    return data

# Menambahkan indikator teknikal
def add_technical_indicators(data):
    data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    return data

# Melatih model AI
def train_ai_model(data):
    data = data.dropna()
    if data.empty:
        raise ValueError("Data setelah penambahan indikator teknikal kosong.")
    
    X = data[['SMA', 'RSI']]
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    if X.empty or y.empty:
        raise ValueError("Data fitur atau label kosong.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if X_train.size == 0 or X_test.size == 0:
        raise ValueError("Dataset pelatihan atau pengujian kosong setelah train_test_split.")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    return model, scaler

# Mengecek saldo akun sebelum melakukan trading dan menyesuaikan kuantitas
def check_balance(symbol, quantity, action, client):
    asset = symbol.replace("USDT", "")
    balances = client.get_asset_balance(asset=asset)
    if balances is None:
        raise ValueError(f"Tidak dapat mengambil saldo untuk {asset}.")
    
    available_balance = float(balances['free'])
    
    if available_balance < quantity:
        if action == 'Sell':
            print(f"Saldo tidak mencukupi untuk menjual {quantity} {asset}. Menyesuaikan jumlah menjadi {available_balance} {asset}.")
            quantity = available_balance
        else:
            raise ValueError(f"Saldo tidak mencukupi untuk membeli {quantity} {asset}. Saldo saat ini: {available_balance} {asset}")
    
    return quantity

# Menyesuaikan kuantitas agar sesuai dengan LOT_SIZE
def adjust_quantity_to_lot_size(symbol, quantity, client):
    exchange_info = client.get_symbol_info(symbol)
    lot_size_info = next(filter(lambda x: x['filterType'] == 'LOT_SIZE', exchange_info['filters']), None)
    
    if lot_size_info is not None:
        min_qty = float(lot_size_info['minQty'])
        max_qty = float(lot_size_info['maxQty'])
        step_size = float(lot_size_info['stepSize'])
        
        # Pastikan kuantitas sesuai dengan LOT_SIZE
        if quantity < min_qty:
            quantity = min_qty
        elif quantity > max_qty:
            quantity = max_qty
        else:
            # Menyesuaikan kuantitas agar sesuai dengan step_size
            quantity = (quantity // step_size) * step_size
        
        quantity = round(quantity, len(str(step_size).split('.')[1]))
    
    return quantity

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
    quantity = check_balance(symbol, quantity, action, client)
    quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
    
    try:
        if action == 'Buy':
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
        elif action == 'Sell':
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
        
        print(f"Executed {action} order: {order}")
        return order
    except BinanceAPIException as e:
        print(f"Error saat eksekusi trade: {e}")
        return None

# Menjalankan bot trading
def run_trading_bot(market_symbol, trade_symbol, quantity, client):
    try:
        data = get_market_data(market_symbol)
        data = add_technical_indicators(data)
        
        model, scaler = train_ai_model(data)
        
        action = make_trade_decision(data, model, scaler)
        
        execute_trade(action, trade_symbol, quantity, client)
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

# Loop untuk menjalankan bot secara otomatis
def start_bot(market_symbol, trade_symbol, quantity, client, interval=3600):
    while True:
        print("Memulai siklus trading...")
        run_trading_bot(market_symbol, trade_symbol, quantity, client)
        print(f"Menunggu {interval / 60} menit sebelum siklus berikutnya...")
        time.sleep(interval)

# Sampel eksekusi
if __name__ == "__main__":
    client = Client('h6js6UiH8EDXBRhzQYWoYUjBxEisuf0OgD86BD6bcfrn2UAvx7sYBShd8LIoOj2a', 'Sg6yoywPejPggWekj40oGHz1vQivrg5tNoSXyWVFcsqPgUmcxCEbUjvI1KyOg1TS')
    # Menjalankan bot dengan simbol yang benar untuk yfinance dan Binance untuk DOGE/USDT
    start_bot('DOGE-USD', 'DOGEUSDT', 100, client)
