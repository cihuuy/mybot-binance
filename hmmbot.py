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
from hmmlearn import hmm

# Function to get market data from Binance API
def get_market_data(symbol, client, interval='1h', limit=1000):
    print(f"Downloading market data for {symbol}...")
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
    print(f"Data downloaded: {df.tail()}")
    return df

# Function to add technical indicators to market data
def add_technical_indicators(data):
    print("Adding technical indicators...")
    data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    data['MACD'], data['MACD_SIGNAL'], data['MACD_HIST'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    return data

# Function to train Gradient Boosting model with RandomizedSearchCV and TimeSeriesSplit
def train_gradient_boosting_model(data):
    print("Training Gradient Boosting model...")
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

    with open('gradient_boosting_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)
    with open('gradient_boosting_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    return best_model, scaler, model_accuracy

# Function to train HMM model
def train_hmm(data, n_components):
    print("Training HMM model...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    model = hmm.GaussianHMM(n_components=n_components)
    model.fit(data_scaled)
    return model, scaler

# Function to predict using HMM
def predict_hmm(model, scaler, data):
    print("Predicting hidden states...")
    data_scaled = scaler.transform(data)
    hidden_states = model.predict(data_scaled)
    state_probs = model.predict_proba(data_scaled)
    return hidden_states, state_probs

# Function to make decision based on HMM
def make_decision_hmm(hidden_states, state_probs, data):
    latest_state_probs = state_probs[-1]
    state_labels = ['State 0', 'State 1', 'State 2']  # Adjust according to n_components
    max_prob_state = np.argmax(latest_state_probs)
    action = 'Hold'
    
    if latest_state_probs[max_prob_state] > 0.6:
        if max_prob_state == 0:
            action = 'Buy'
        elif max_prob_state == 1:
            action = 'Sell'
    
    state_probs_summary = dict(zip(state_labels, latest_state_probs))
    print(f"Latest State Probabilities: {state_probs_summary}")
    print(f"Trade decision: {action}")
    
    return action, state_probs_summary

# Function to check balance and adjust quantity
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

# Function to adjust quantity to match LOT_SIZE
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

# Function to execute trade order
def execute_trade(action, symbol, quantity, client, stop_loss=None, take_profit=None):
    quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
    print(f"Executing {action} trade for {symbol} with quantity {quantity:.6f}")
    try:
        if action == 'Buy':
            order = client.order_market_buy(symbol=symbol, quantity=quantity)
            print(f"Buy order placed: {order}")
            if stop_loss is not None and take_profit is not None:
                limit_price = (stop_loss + take_profit) / 2
                stop_price = stop_loss
                client.create_oco_order(symbol=symbol, side='SELL', quantity=quantity, price=limit_price, stopPrice=stop_price, stopLimitPrice=stop_price, stopLimitTimeInForce='GTC')
                print(f"OCO order placed with Stop-Loss at {stop_loss:.6f} and Take-Profit at {take_profit:.6f}")
        elif action == 'Sell':
            available_balance = client.get_asset_balance(asset=symbol.replace("USDT", ""))
            if available_balance is None:
                print(f"Tidak ditemukan saldo untuk {symbol.replace('USDT', '')}.")
                return
            available_quantity = float(available_balance['free'])
            if quantity > available_quantity:
                quantity = available_quantity
            order = client.order_market_sell(symbol=symbol, quantity=quantity)
            print(f"Sell order placed: {order}")
    except BinanceAPIException as e:
        print(f"Error executing trade: {e.message}")

# Main trading function
def trade(symbol, client, model_gb, scaler_gb, model_hmm, scaler_hmm, n_components, quantity=10):
    df = get_market_data(symbol, client)
    df = add_technical_indicators(df)

    # Gradient Boosting Prediction
    X = df[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']].dropna()
    X = scaler_gb.transform(X)
    gb_prediction = model_gb.predict(X[-1].reshape(1, -1))[0]

    # HMM Prediction
    hmm_data = df[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']].dropna()
    hidden_states, state_probs = predict_hmm(model_hmm, scaler_hmm, hmm_data)
    action_hmm, state_probs_summary = make_decision_hmm(hidden_states, state_probs, hmm_data)

    action = 'Hold'
    if gb_prediction == 1 and action_hmm == 'Buy':
        action = 'Buy'
    elif gb_prediction == 0 and action_hmm == 'Sell':
        action = 'Sell'

    quantity = check_balance(symbol, quantity, action, client)
    stop_loss = df['Low'].min()
    take_profit = df['High'].max()

    execute_trade(action, symbol, quantity, client, stop_loss, take_profit)

if __name__ == "__main__":
    API_KEY = 'h6js6UiH8EDXBRhzQYWoYUjBxEisuf0OgD86BD6bcfrn2UAvx7sYBShd8LIoOj2a'
    API_SECRET = 'Sg6yoywPejPggWekj40oGHz1vQivrg5tNoSXyWVFcsqPgUmcxCEbUjvI1KyOg1TS'
    client = Client(API_KEY, API_SECRET)

    # Load Gradient Boosting model and scaler
    with open('gradient_boosting_model.pkl', 'rb') as model_file:
        model_gb = pickle.load(model_file)
    with open('gradient_boosting_scaler.pkl', 'rb') as scaler_file:
        scaler_gb = pickle.load(scaler_file)

    # Load HMM model and scaler
    with open('hmm_model.pkl', 'rb') as model_file:
        model_hmm = pickle.load(model_file)
    with open('hmm_scaler.pkl', 'rb') as scaler_file:
        scaler_hmm = pickle.load(scaler_file)

    SYMBOL = 'DOGEUSDT'
    N_COMPONENTS = 3  # Adjust based on your model training
    TRADE_QUANTITY = 10

    while True:
        try:
            trade(SYMBOL, client, model_gb, scaler_gb, model_hmm, scaler_hmm, N_COMPONENTS, TRADE_QUANTITY)
            time.sleep(3600)  # Wait for 1 hour before checking again
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(60)  # Wait for a minute before retrying
