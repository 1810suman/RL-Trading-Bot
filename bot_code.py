import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from alpha_vantage.timeseries import TimeSeries

# ==============================
# ✅ Step 1: Fetch & Save Stock Data
# ==============================
API_KEY = "YOUR API KEY"  # Consider using environment variables for API keys
DATA_FILES = {"AAPL": "aapl_stock_data.csv", "GOOGL": "googl_stock_data.csv", "MSFT": "msft_stock_data.csv"}

def fetch_stock_data(symbol, interval="5min"):
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    data, _ = ts.get_intraday(symbol=symbol, interval=interval, outputsize="full")
    data.columns = ["Open", "High", "Low", "Close", "Volume"]
    data.to_csv(DATA_FILES[symbol])
    return data

# Load or fetch data
dfs = {}
for symbol in DATA_FILES.keys():
    if not os.path.exists(DATA_FILES[symbol]):
        print(f"Fetching stock data for {symbol}...")
        dfs[symbol] = fetch_stock_data(symbol)
    else:
        dfs[symbol] = pd.read_csv(DATA_FILES[symbol], index_col=0)
        dfs[symbol] = dfs[symbol][['Open', 'High', 'Low', 'Close', 'Volume']]
        print(f"Loaded existing stock data for {symbol}.")

# Compute Technical Indicators
def compute_indicators(df):
    df = df.copy()  # Avoid SettingWithCopyWarning
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    # Calculate RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # Calculate MACD (Moving Average Convergence Divergence)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    # Fill NaN values using more appropriate method for time series
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

dfs = {symbol: compute_indicators(df) for symbol, df in dfs.items()}

# ==============================
# ✅ Step 2: Define Custom Trading Environment
# ==============================
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee_percent = transaction_fee_percent
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_steps = len(df) - 1
        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32)
        self.trading_log = []
        
    def normalize_observation(self, observation):
        # Scale the features to help the model learn better
        return (observation - np.mean(observation)) / (np.std(observation) + 1e-8)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["Close"]
        previous_net_worth = self.net_worth
        trade = {"Step": self.current_step, "Action": action, "Price": current_price}

        # Apply transaction costs for realistic simulation
        if action == 1 and self.balance >= current_price:  # Buy
            transaction_cost = current_price * self.transaction_fee_percent
            max_shares_possible = self.balance // (current_price + transaction_cost)
            
            if max_shares_possible > 0:
                self.shares_held += 1
                self.balance -= (current_price + transaction_cost)
                trade["Shares"] = 1
                trade["Cost"] = current_price + transaction_cost
        
        elif action == 2 and self.shares_held > 0:  # Sell
            transaction_cost = current_price * self.transaction_fee_percent
            self.balance += (current_price - transaction_cost)
            self.shares_held -= 1
            trade["Shares"] = -1
            trade["Revenue"] = current_price - transaction_cost

        self.trading_log.append(trade)
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # Calculate reward - improved reward function
        reward = (self.net_worth - previous_net_worth) / previous_net_worth * 100  # Percentage return
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Additional info dictionary for debugging
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'reward': reward
        }
        
        # Normalize observation for better learning
        next_obs = self.normalize_observation(self.df.iloc[self.current_step].values.astype(np.float32))
        
        return next_obs, reward, done, False, info  # Additional False for truncated parameter in gym v26+

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Properly handle the seed
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.trading_log = []
        
        # Return normalized initial observation
        initial_obs = self.normalize_observation(self.df.iloc[self.current_step].values.astype(np.float32))
        # In Gymnasium v26+, reset returns (observation, info)
        return initial_obs, {}

    def get_trading_log(self):
        return pd.DataFrame(self.trading_log)

# ==============================
# ✅ Step 3: Train RL Model (PPO)
# ==============================
print("Training PPO agent...")

models = {}
training_histories = {}

for symbol, df in dfs.items():
    env = TradingEnv(df)
    env = DummyVecEnv([lambda: env])
    MODEL_PATH = f"ppo_trading_bot_{symbol}.zip"
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model for {symbol}...")
        model = PPO.load(MODEL_PATH, env)
    else:
        print(f"Training new model for {symbol}...")
        # Better hyperparameters
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        history = model.learn(total_timesteps=100000)
        training_histories[symbol] = history
        model.save(MODEL_PATH)
    
    models[symbol] = model
    print(f"✅ Training Complete for {symbol}! Model saved.")

# ==============================
# ✅ Step 4: Test the Model & Visualize Results
# ==============================
fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
all_results = {}

for i, (symbol, df) in enumerate(dfs.items()):
    print(f"Testing trained model for {symbol}...")
    env = TradingEnv(df)
    obs, _ = env.reset()
    net_worths = []
    actions = []
    balances = []
    shares_held = []
    prices = df["Close"].tolist()

    for _ in range(len(df) - 1):
        action, _ = models[symbol].predict(obs)
        actions.append(action)
        obs, _, done, _, info = env.step(action)
        
        net_worths.append(info['net_worth'])
        balances.append(info['balance'])
        shares_held.append(info['shares_held'])
        
        if done:
            break

    all_results[symbol] = {
        'net_worths': net_worths,
        'actions': actions,
        'prices': prices[:len(net_worths)],
        'balances': balances,
        'shares_held': shares_held
    }
    
    # Plot net worth and stock price
    axes[i].plot(net_worths, label=f"{symbol} Net Worth", linewidth=2)
    axes[i].plot(prices[:len(net_worths)], label=f"{symbol} Stock Price", linestyle='dashed')
    
    # Plot buy and sell actions
    buy_indices = [j for j in range(len(actions)) if actions[j] == 1]
    sell_indices = [j for j in range(len(actions)) if actions[j] == 2]
    
    axes[i].scatter(buy_indices, [prices[j] for j in buy_indices], 
                 label=f"{symbol} Buy", color="green", marker="^", s=100)
    axes[i].scatter(sell_indices, [prices[j] for j in sell_indices], 
                  label=f"{symbol} Sell", color="red", marker="v", s=100)
    
    axes[i].set_title(f"{symbol} Trading Performance")
    axes[i].set_ylabel("Value ($)")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

axes[-1].set_xlabel("Time Steps")
plt.tight_layout()
plt.savefig("trading_results.png")  # Save the figure before showing
plt.show()

# Calculate performance metrics
for symbol, results in all_results.items():
    initial_value = results['net_worths'][0]
    final_value = results['net_worths'][-1]
    total_return = ((final_value - initial_value) / initial_value) * 100
    
    # Buy and hold strategy returns
    buy_hold_return = ((results['prices'][-1] - results['prices'][0]) / results['prices'][0]) * 100
    
    # Count trades
    buy_trades = results['actions'].count(1)
    sell_trades = results['actions'].count(2)
    
    print(f"\n{symbol} Performance Metrics:")
    print(f"Starting Value: ${initial_value:.2f}")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Total Trades: {buy_trades + sell_trades} (Buy: {buy_trades}, Sell: {sell_trades})")
    print(f"Outperformance vs Buy & Hold: {total_return - buy_hold_return:.2f}%")
