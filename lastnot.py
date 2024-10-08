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
from datetime import datetime
from scipy.stats import norm

# Get market data from Binance API
def get_market_data(symbol, client, interval='15m', limit=1000):
    print(f"Downloading market data for {symbol} using Binance API...")
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
    print(f"Data downloaded: {df.head()}")
    return df

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
    print(f"Model accuracy: {model_accuracy * 100:.2f}%")
    print("Best parameters found: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)

    # Save model and scaler
    with open('trading_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    return best_model, scaler, model_accuracy

# Backtest the model
def backtest_model(data, model, scaler):
    print("Backtesting model...")
    data = data.dropna()
    X = data[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']]
    y = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    backtest_accuracy = accuracy_score(y, y_pred)
    print(f"Backtest accuracy: {backtest_accuracy * 100:.2f}%")
    return backtest_accuracy

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

# Make trading decision and determine stop-loss and take-profit
def make_trade_decision(data, model, scaler):
    print("Making trade decision...")
    X = data[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']].dropna()
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled[-1:])
    stop_loss_pct = model.predict_proba(X_scaled[-1:])[0][1] * 0.02  # Example multiplier for risk adjustment
    take_profit_pct = model.predict_proba(X_scaled[-1:])[0][0] * 0.02  # Example multiplier for reward adjustment
    current_price = data['Close'].iloc[-1]
    stop_loss = current_price * (1 - stop_loss_pct)
    take_profit = current_price * (1 + take_profit_pct)
    action = 'Buy' if prediction == 1 else 'Sell'
    print(f"Trade decision: {action}")
    print(f"Calculated Stop-Loss: {stop_loss:.6f}")
    print(f"Calculated Take-Profit: {take_profit:.6f}")
    return action, stop_loss, take_profit

# Execute trade order
def execute_trade(action, symbol, quantity, client, stop_loss=None, take_profit=None):
    # Adjust quantity to meet Binance's LOT_SIZE requirement
    quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
    
    print(f"Executing {action} trade for {symbol} with quantity {quantity:.6f}")
    try:
        if action == 'Buy':
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
            print(f"Buy order placed: {order}")
            if stop_loss is not None and take_profit is not None:
                limit_price = (stop_loss + take_profit) / 2  # Example limit price
                stop_price = stop_loss
                client.create_oco_order(symbol=symbol, side='SELL', quantity=quantity, price=limit_price, stopPrice=stop_price, stopLimitPrice=stop_price, stopLimitTimeInForce='GTC')
                print(f"OCO order placed with Stop-Loss at {stop_loss:.6f} and Take-Profit at {take_profit:.6f}")
        elif action == 'Sell':
            quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
            available_balance = client.get_asset_balance(asset=symbol.replace("USDT", ""))
            if available_balance is None:
                print(f"Tidak ditemukan saldo untuk {symbol.replace('USDT', '')}.")
                return
            available_quantity = float(available_balance['free'])
            if available_quantity < quantity:
                print(f"Tidak cukup saldo {symbol.replace('USDT', '')} untuk menjual {quantity:.6f}. Menyesuaikan jumlah menjadi {available_quantity:.6f}.")
                quantity = available_quantity
                quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
            if quantity <= 0:
                print(f"Tidak ada jumlah yang valid untuk menjual.")
                return
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
            print(f"Sell order placed: {order}")
        else:
            print(f"Unknown action {action}. Skipping trade.")
    except BinanceAPIException as e:
        print(f"Binance API Exception: {e}")
    except Exception as e:
        print(f"Error executing trade: {e}")

# Main function to run the trading strategy
def main():
    # Initialize Binance client
    api_key = 'h6js6UiH8EDXBRhzQYWoYUjBxEisuf0OgD86BD6bcfrn2UAvx7sYBShd8LIoOj2a'
    api_secret = 'Sg6yoywPejPggWekj40oGHz1vQivrg5tNoSXyWVFcsqPgUmcxCEbUjvI1KyOg1TS'
    client = Client(api_key, api_secret)

    # Define trading symbol
    symbol = 'DOGEUSDT'

    while True:
        try:
            # Get market data
            df = get_market_data(symbol, client)
            df = add_technical_indicators(df)

            # Load model and scaler or train if not available
            model, scaler = load_model_and_scaler()
            if model is None or scaler is None:
                model, scaler, _ = train_ai_model(df)

            # Backtest the model
            backtest_accuracy = backtest_model(df, model, scaler)
            print(f"Backtest accuracy: {backtest_accuracy * 100:.2f}%")

            # Make trading decision
            action, stop_loss, take_profit = make_trade_decision(df, model, scaler)
            
            # Determine the quantity to trade (example quantity)
            quantity = 1.0

            # Check balance and adjust quantity
            quantity = check_balance(symbol, quantity, action, client)

            # Execute trade
            execute_trade(action, symbol, quantity, client, stop_loss, take_profit)

            # Print model accuracy
            model_accuracy = accuracy_score(
                df[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']].dropna(),
                model.predict(scaler.transform(df[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']].dropna()))
            )
            print(f"Model accuracy during trading: {model_accuracy * 100:.2f}%")

        except Exception as e:
            print(f"Error in main loop: {e}")

        # Wait for 15 minutes before the next iteration
        time.sleep(15 * 60)

if __name__ == "__main__":
    main()
