import yfinance as yf
import talib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
import pickle
from datetime import datetime, timedelta
from scipy.stats import norm

# Get market data from Binance API
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

# Add technical indicators to market data
def add_technical_indicators(data):
    print("Adding technical indicators...")
    data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_SIGNAL'], data['MACD_HIST'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    return data

# Train AI model with RandomizedSearchCV and TimeSeriesSplit
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
    
    param_distributions = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions, n_iter=10, cv=tscv, scoring='accuracy', random_state=42)
    random_search.fit(X_train, y_train)
    
    model = random_search.best_estimator_
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # Save model and scaler
    with open('trading_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    
    return model, scaler, X_test, y_test, accuracy

# Load saved model and scaler
def load_model_and_scaler():
    try:
        with open('trading_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except FileNotFoundError:
        return None, None

# Check account balance before trading and adjust quantity
def check_balance(symbol, quantity, action, client):
    asset = symbol.replace("USDT", "")
    
    if action == 'Buy':
        print(f"Checking balance for USDT...")
        balances = client.get_asset_balance(asset='USDT')
        if balances is None:
            raise ValueError("Tidak dapat mengambil saldo untuk USDT.")
        
        available_balance = float(balances['free'])
        price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        required_balance = quantity * price

        if available_balance < required_balance:
            quantity = available_balance / price
            print(f"Saldo tidak mencukupi untuk membeli {quantity:.6f} {symbol}. Menyesuaikan jumlah menjadi {quantity:.6f} {symbol}.")
        
        quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
    
    elif action == 'Sell':
        print(f"Checking balance for {asset}...")
        balances = client.get_asset_balance(asset=asset)
        if balances is None:
            raise ValueError(f"Tidak dapat mengambil saldo untuk {asset}.")
        
        available_balance = float(balances['free'])
        
        if available_balance < quantity:
            print(f"Saldo tidak mencukupi untuk menjual {quantity} {asset}. Menyesuaikan jumlah menjadi {available_balance} {asset}.")
            quantity = available_balance
    
    return quantity

# Adjust quantity to match LOT_SIZE
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

# Calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.95):
    mean_return = np.mean(returns)
    std_dev = np.std(returns)
    var = norm.ppf(1 - confidence_level, mean_return, std_dev)
    return var

# Make trading decision
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

# Execute trade order with stop-loss and take-profit
def execute_trade(action, symbol, quantity, client, stop_loss=None, take_profit=None):
    print(f"Executing {action} trade...")
    if action == 'Buy':
        try:
            quantity = check_balance(symbol, quantity, action, client)
            if quantity is None:
                print("Tidak cukup saldo USDT untuk membeli.")
                return
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
            print(f"Executed Buy order: {order}")

            if stop_loss:
                stop_loss_order = client.create_order(
                    symbol=symbol,
                    side='SELL',
                    type='STOP_MARKET',
                    quantity=quantity,
                    stopPrice=stop_loss
                )
                print(f"Stop-loss order placed: {stop_loss_order}")

            if take_profit:
                take_profit_order = client.create_order(
                    symbol=symbol,
                    side='SELL',
                    type='TAKE_PROFIT_MARKET',
                    quantity=quantity,
                    stopPrice=take_profit
                )
                print(f"Take-profit order placed: {take_profit_order}")

        except BinanceAPIException as e:
            print(f"Error saat eksekusi trade: {e}")
                
    elif action == 'Sell':
        try:
            quantity = check_balance(symbol, quantity, action, client)
            if quantity is None:
                print("Tidak cukup saldo untuk menjual.")
                return
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
            print(f"Executed Sell order: {order}")

        except BinanceAPIException as e:
            if e.code == -1013:  # Handle insufficient funds error
                print("Tidak cukup saldo untuk melakukan penjualan. Menunggu untuk siklus berikutnya.")
            else:
                print(f"Error saat eksekusi trade: {e}")

# Function to evaluate model accuracy
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Backtest accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Main function to run trading bot
def run_trading_bot(api_key, api_secret, symbol, trade_quantity, retrain_interval_days=1):
    client = Client(api_key=api_key, api_secret=api_secret)
    
    # Verify server time before starting trading process
    server_time = datetime.fromtimestamp(client.get_server_time()['serverTime'] / 1000.0)
    print(f"Server time: {server_time}")
    
    model, scaler = load_model_and_scaler()
    last_training_date = datetime.now() - timedelta(days=retrain_interval_days)
    
    if model is None or scaler is None or datetime.now() - last_training_date > timedelta(days=retrain_interval_days):
        print("Training new model...")
        data = get_market_data(symbol, client, '1h', 1000)
        data = add_technical_indicators(data)
        model, scaler, X_test, y_test, model_accuracy = train_ai_model(data)
        backtest_accuracy = evaluate_model(model, X_test, y_test)
        if model_accuracy >= 0.35 and backtest_accuracy >= 0.35:
            print("Model trained successfully and meets accuracy requirements.")
            last_training_date = datetime.now()  # Update the last training date
        else:
            print("Model accuracy or backtest accuracy is below the threshold. Skipping trading.")
            # Print a placeholder decision
            try:
                data = get_market_data(symbol, client, '1h', 1000)
                data = add_technical_indicators(data)
                action = make_trade_decision(data, model, scaler)
                print(f"Trade decision (skipped trading): {action}")
            except ValueError as e:
                print(f"An error occurred while making trade decision: {e}")
            
            # Continue running the bot without trading
            while True:
                market_time = datetime.fromtimestamp(client.get_server_time()['serverTime'] / 1000.0)
                print(f"Market time: {market_time}")
                time.sleep(60 * 60)  # Sleep for 1 hour
            return
    
    while True:
        market_time = datetime.fromtimestamp(client.get_server_time()['serverTime'] / 1000.0)
        print(f"Market time: {market_time}")
        
        try:
            data = get_market_data(symbol, client, '1h', 1000)
            data = add_technical_indicators(data)
            action = make_trade_decision(data, model, scaler)
            
            # Define stop-loss and take-profit levels
            current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
            stop_loss = current_price * 0.98  # Example: 2% below current price
            take_profit = current_price * 1.02  # Example: 2% above current price
            
            execute_trade(action, symbol, trade_quantity, client, stop_loss, take_profit)
        except ValueError as e:
            print(f"An error occurred: {e}")
        
        time.sleep(60 * 60)  # Sleep for 1 hour

if __name__ == "__main__":
    # Replace these with your actual API keys and trading symbol
    api_key = 'h6js6UiH8EDXBRhzQYWoYUjBxEisuf0OgD86BD6bcfrn2UAvx7sYBShd8LIoOj2a'
    api_secret = 'Sg6yoywPejPggWekj40oGHz1vQivrg5tNoSXyWVFcsqPgUmcxCEbUjvI1KyOg1TS'
    symbol = 'DOGEUSDT'
    trade_quantity = 100  # Adjust trade quantity as needed
    
    run_trading_bot(api_key, api_secret, symbol, trade_quantity)
