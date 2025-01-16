import torch
import os
import ccxt
import pandas as pd
import time
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
import smtplib
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from data.crypto_dataset import CryptoDataset
from data.data_preprocessor import DataPreprocessor
from data.feature_engineering import FeatureEngineering
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.ensemble_model import EnsembleModel
from trainers.trainer_lstm import TrainerLSTM
from trainers.trainer_transformer import TrainerTransformer
from trainers.trainer_ensemble import TrainerEnsemble
from utils.predict import predict_signal
from utils.signal_generator import generate_signal
from utils.twitter_client import tweet_signal
from utils.config import Config
from utils.logger import Logger
from agents.agent_factory import AgentFactory
from agents.agent_manager import AgentManager

api_key = "API_KEY"
api_secret = "API_SECRET"

exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})

symbol = 'BTC/USDT'  # Market pair to trade
timeframe = '1h'  # Timeframe for candles
sma_short = 7  # Short-term SMA
sma_long = 25  # Long-term SMA

# Risk management settings
stop_loss_pct = 0.02  # 2% stop-loss
take_profit_pct = 0.05  # 5% take-profit

# Position tracking
position = {
    "type": None,  # 'buy' or 'sell'
    "entry_price": None,
    "amount": None
}


def fetch_data():
    """Fetch OHLCV data from the exchange."""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def calculate_sma(data, window):
    """Calculate Simple Moving Average (SMA)."""
    return data['close'].rolling(window=window).mean()


def check_signals(data):
    """Check buy/sell signals based on SMA crossovers."""
    if data['sma_short'].iloc[-1] > data['sma_long'].iloc[-1] and \
       data['sma_short'].iloc[-2] <= data['sma_long'].iloc[-2]:
        return 'buy'
    elif data['sma_short'].iloc[-1] < data['sma_long'].iloc[-1] and \
         data['sma_short'].iloc[-2] >= data['sma_long'].iloc[-2]:
        return 'sell'
    return None


def place_order(signal, tolerance=0.01):
    """
    Place a buy or sell order. Avoid execution if price moves beyond tolerance.
    """
    global position
    ticker = exchange.fetch_ticker(symbol)
    last_price = ticker['last']

    balance = exchange.fetch_balance()

    # Handle buy signal
    if signal == 'buy' and position['type'] is None:
        amount = balance['USDT']['free'] / last_price
        exchange.create_market_buy_order(symbol, amount)
        position = {"type": "buy", "entry_price": last_price, "amount": amount}
        print(f"Bought {amount} of {symbol} at {last_price}")

    # Handle sell signal
    elif signal == 'sell' and position['type'] == 'buy':
        amount = position['amount']
        exchange.create_market_sell_order(symbol, amount)
        print(f"Sold {amount} of {symbol} at {last_price}")
        position = {"type": None, "entry_price": None, "amount": None}


def manage_risk():
    """Manage stop-loss and take-profit for open positions."""
    global position
    if position['type'] == 'buy':
        ticker = exchange.fetch_ticker(symbol)
        last_price = ticker['last']

        stop_loss_price = position['entry_price'] * (1 - stop_loss_pct)
        take_profit_price = position['entry_price'] * (1 + take_profit_pct)

        if last_price <= stop_loss_price:
            print(f"Stop-loss triggered at {last_price}")
            place_order('sell')
        elif last_price >= take_profit_price:
            print(f"Take-profit reached at {last_price}")
            place_order('sell')


def send_email(subject, body):
    """Send an email notification."""
    sender = "your_email@example.com"
    recipient = "recipient_email@example.com"
    password = "YOUR_EMAIL_PASSWORD"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
        print("Email sent successfully")


# Main loop
while True:
    try:
        # Fetch and process data
        data = fetch_data()
        data['sma_short'] = calculate_sma(data, sma_short)
        data['sma_long'] = calculate_sma(data, sma_long)

        if position['type'] is None:  # If no open position, check for signals
            signal = check_signals(data)
            if signal:
                place_order(signal)
        else:  # Manage existing position
            manage_risk()

        # Wait for the next candle
        time.sleep(60 * 60)

    except Exception as e:
        print(f"Error: {e}")
        send_email("Trading Bot Error", f"An error occurred: {e}")
        time.sleep(60)
