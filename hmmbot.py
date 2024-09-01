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
    with open('gradient_boosting_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)
    with open('gradient_boosting_scaler.pkl', 'wb') as scaler_file:
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
        with open('gradient_boosting_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('gradient_boosting_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except FileNotFoundError:
        print("Model or scaler file not found. Training a new model.")
        return None, None

# Train HMM model
def train_hmm(data, n_components):
    print("Training HMM model...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    model = hmm.GaussianHMM(n_components=n_components)
    model.fit(data_scaled)
    return model, scaler

# Predict with HMM model
def predict_hmm(model, scaler, data):
    print("Predicting hidden states with HMM...")
    data_scaled = scaler.transform(data)
    hidden_states = model.predict(data_scaled)
    
    # Probabilities of hidden states
    state_probs = model.predict_proba(data_scaled)
    return hidden_states, state_probs

# Make trading decision
def make_trade_decision(data, model, scaler, use_hmm=False):
    if use_hmm:
        # HMM decision
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
        print(f"Latest State Probabilities: {state_probs_summary}")
        print(f"Trade decision (HMM): {action}")
        return action, state_probs_summary
    
    else:
        # AI decision
        X = data[['SMA', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'BB_upper', 'BB_middle', 'BB_lower', 'ATR']].dropna()
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled[-1:])
        stop_loss_pct = model.predict_proba(X_scaled[-1:])[0][1] * 0.02  # Example multiplier for risk adjustment
        take_profit_pct = model.predict_proba(X_scaled[-1:])[0][0] * 0.02  # Example multiplier for reward adjustment
        current_price = data['Close'].iloc[-1]
        stop_loss = current_price * (1 - stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct)
        action = 'Buy' if prediction == 1 else 'Sell'
        print(f"Trade decision (AI): {action}")
        print(f"Calculated Stop-Loss: {stop_loss:.6f}")
        print(f"Calculated Take-Profit: {take_profit:.6f}")
        return action, stop_loss, take_profit

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
        quantity = round(quantity, len(str(step_size).split('.')[1]))  # Match precision of step size
    return quantity

# Place a market order
def place_market_order(symbol, side, quantity, client):
    try:
        adjusted_quantity = adjust_quantity_to_lot_size(symbol, quantity, client)
        print(f"Placing {side} market order for {symbol} with quantity {adjusted_quantity:.6f}...")
        order = client.order_market(
            symbol=symbol,
            side=side,
            quantity=adjusted_quantity
        )
        print("Order placed successfully.")
        return order
    except BinanceAPIException as e:
        print(f"Binance API exception occurred: {e}")
        return None

# Execute trading strategy
def execute_trading_strategy():
    client = Client(api_key='h6js6UiH8EDXBRhzQYWoYUjBxEisuf0OgD86BD6bcfrn2UAvx7sYBShd8LIoOj2a', api_secret='Sg6yoywPejPggWekj40oGHz1vQivrg5tNoSXyWVFcsqPgUmcxCEbUjvI1KyOg1TS')
    symbol = 'DOGEUSDT'
    df = get_market_data(symbol, client)
    df = add_technical_indicators(df)
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        model, scaler, _ = train_ai_model(df)
    
    action, stop_loss, take_profit = make_trade_decision(df, model, scaler, use_hmm=False)
    
    # Define the quantity for trading
    quantity = 100  # Example quantity in DOGE
    
    if action == 'Buy':
        place_market_order(symbol, 'BUY', quantity, client)
    elif action == 'Sell':
        place_market_order(symbol, 'SELL', quantity, client)

    # Add your trading logic here (e.g., stop loss, take profit)

# Run the trading strategy every hour
while True:
    execute_trading_strategy()
    time.sleep(3600)  # Wait for 1 hour
