# Divon 
Divon is an framework for analyzing and trading cryptocurrency markets generating AI-driven trading signals, and broadcasting these insights to platforms like Twitter. Built with a rich feature set, it leverages cutting-edge AI models (LSTM, Transformers, and custom Ensembles), robust data pipelines, and an extensible agent framework for tailored trading strategies. 

---

## Divon is powered by `divon` Solana token with contract address :

#### CA:  `4xsyS5vGxRCH2rhYTNf99VpeVamxJ2A4hFzxdgUmpump`

#### Dev: `6u9zAaSnGgTWRXZWAE1hDL2VUidyPrCzs2J3xeHpUjiZ`

 -Website: [divon.cloud](https://divon.cloud/) 
 
 -X(Twitter): [@DivonApp](https://x.com/DivonApp/) 

<img src="./tests/Divon2_2.png" width=100% height=400>

## Overview

This repository provides a complete pipeline for building, training, testing, and deploying AI models to generate actionable cryptocurrency trading signals. By utilizing PyTorch for deep learning, CCXT for market data acquisition, and Tweepy for social media integration, Divon offers a scalable ecosystem for end-to-end AI-powered trading solutions.

## Features

### **Data Handling**
- Pulls market data via CCXT and enriches it through advanced statistical transformations.
- Supports rolling averages, volatility measures, and returns calculations to enhance feature engineering.
- Provides a structured dataset class for efficient training workflows.

### **AI Models**
- Implements multiple architectures:
  - **LSTM**: Sequential data processing for capturing time-series dependencies.
  - **Transformers**: Attention-based modeling for advanced sequence analysis.
  - **Custom Ensembles**: Combines models to reduce noise and improve signal reliability.

### **Agent Framework**
- Build and manage customizable AI agents, each with unique hyperparameters and configurations.
- Supports multiple strategies and modular workflows under a unified framework.

### **Training and Evaluation**
- Includes configurable trainer classes for streamlined model training.
- Provides robust evaluation scripts for metrics such as MSE, RMSE, and MAE.
- Integrated backtesting capabilities simulate historical performance.

### **Twitter Integration**
- Automatically posts trading signals to Twitter for real-time updates.
- Modular broadcasting system allows easy integration with alternative platforms.

### **Testing and CI**
- Comprehensive Pytest suite ensures stability and reliability.
- Validates data integrity, model architecture, training routines, and utilities.

### **Logging and Auditing**
- Production-ready logging for training progress, trading signals, and system events.
- Facilitates debugging, compliance, and auditing purposes.

---

## Key Features

### **Neural Network Architectures**
- Multiple models designed for time-series and sequential data.
- Ensemble approaches for robust, low-variance signals.

### **Sophisticated Data Pipeline**
- Dynamic feature engineering with rolling statistics and sequence-based preprocessing.
- Automated data validation and cleaning.

### **Customizable AI Agents**
- Define, configure, and manage agents with unique trading strategies.
- Persistent storage of configurations for reproducibility.

### **Backtesting & Evaluation**
- Historical backtesting for assessing strategy effectiveness.
- Pre-built metrics and integration with scikit-learn.

### **Production-Ready Design**
- Modular and extensible for easy integration into larger workflows.
- Local and cloud logging support for regulatory compliance.

---

## **Installation**

1. **Dependencies**
   Installation of the necessary libraries:
     - ccxt: A popular library for accessing multiple exchanges.
     - pandas: To manipulate data.
     - numpy: For mathematical calculations.
     - matplotlib or others to visualize the data.
   ```bash
   pip install ccxt pandas numpy python-dotenv
    ```

3. **Clone the Repository**  
   ```bash
   git clone https://github.com/Oxygen-bot/divon.git
   ```
 
3. **Configuration**
    Specified your key and secret API key.
   
    (Mandatory for any trading program)
        
   ```bash
   # key.env
   API_KEY=your_api_key_here
   API_SECRET=your_api_secret_here
   ```
## **List of Trading Bot Functionalities**

1. Sandbox Mode (Simulation)
  Use the sandbox mode to run the bot without risking real funds.
   ```bash
    exchange.set_sandbox_mode(True)   # Activate simulation mode
   ```
    This utilizes a testing environment provided by Binance or other exchanges to simulate trades.

2. Live Trading Mode
  To use the bot in real trading conditions, deactivate the sandbox mode.
   ```bash
    exchange.set_sandbox_mode(False)  # Deactivate sandbox mode for live trading
   ```
    Make sure your API keys are correctly configured and have the appropriate permissions.

3. Backtesting (Testing on Historical Data)
   Test your strategy on past data to evaluate its performance.
   ```bash
    def backtest(data, sma_short, sma_long, stop_loss_pct, take_profit_pct):
    # Backtesting logic here (provided earlier)

    # Load historical data
    df = pd.read_csv("historical_data.csv")
    
    # Run backtest
    backtest(df, sma_short=7, sma_long=25, stop_loss_pct=0.02, take_profit_pct=0.05)
   ```
    Result: A graphical analysis of capital evolution and performance statistics.

4. Stop-Loss and Take-Profit Management
   The bot monitors open positions and applies limits to minimize losses or secure profits.
  
   Define percentage limits:
  ```bash
  stop_loss_pct = 0.02  # 2% maximum loss
  take_profit_pct = 0.05  # 5% profit target
  ```

   Example in the bot:
   ```bash
  manage_risk()  # Automatically handles stop-loss and take-profit
  ```


6. Parameter Optimization
   Find the best combinations of parameters (SMA, stop-loss, take-profit) using a grid search.
 ```bash
  optimize(
   data=df,
   sma_short_range=range(5, 15),
   sma_long_range=range(20, 50),
   stop_loss_range=[0.01, 0.02, 0.03],
   take_profit_range=[0.03, 0.05, 0.07]
  )
 ```

7. Signal-Based Order Execution
  The bot executes orders based on SMA crossovers (basic strategy).
  Example:
 ```bash
  place_order(signal, tolerance=0.01)  # Allow up to 1% price deviation
 ```

9. Logging and Error Handling
  Monitor errors or bot actions with proper logging.

  Error capture example:
 ```bash
  try:
    # Main bot logic
  except Exception as e:
    print(f"Error: {e}")
    send_email("Trading Bot Error", f"An error occurred: {e}")

 ```

### Data Ingestion
By default, the system fetches data from Binance (BTC/USDT, 1-hour intervals). Modify `get_crypto_data` in `src/main.py` to switch to alternative markets or timeframes.

### Agent Configuration
The `AgentFactory` spawns custom agents with default configurations. Edit or update these parameters to tailor models, training hyperparameters, or custom behaviors (e.g., different LSTM vs. Transformer usage).

### Model Training and Signal Generation
- Run `python src/main.py` to initiate the entire pipeline.
- The code automatically trains the chosen model architecture on the downloaded data and generates signals.
- A final Twitter post is made if credentials are available.

### Backtesting
- Execute `python scripts/backtest.py` or integrate `run_backtest` within your custom workflow.
- Supply a DataFrame of historical prices and a list of signals to simulate hypothetical trades and assess performance.

### Evaluation
- Use `python scripts/evaluate_model.py` or the `evaluate` function to compute MSE on validation/test sets.
- Extend or adapt the script to incorporate advanced metrics or cross-validation strategies.

### Managing Multiple Agents
- Save an agent’s configuration with `AgentManager.save_agent(my_agent)`.
- Load an agent’s settings via `agent_manager.load_agent(agent_name)`.
- Keep track of agent files to pivot between various strategies quickly.

## Project Structure
```
Divon-main/
├── .env.example
├── README.md
├── requirements.txt
├── scripts
│   ├── backtest.py
│   └── evaluate_model.py
├── main.py
├── agents
│   ├── agent_factory.py
│   ├── agent_manager.py
│   └── custom_agent.py
├── data
│   ├── crypto_dataset.py
│   ├── data_preprocessor.py
│   └── feature_engineering.py
├── models
│   ├── lstm_model.py
│   ├── transformer_model.py
│   └── ensemble_model.py
├── trainers
│   ├── trainer_lstm.py
│   ├── trainer_transformer.py
│   └── trainer_ensemble.py
└── utils
    ├── config.py
    ├── logger.py
    ├── metrics.py
    ├── predict.py
    ├── signal_generator.py
    ├── train.py
    └── twitter_client.py
└── tests
    ├── test_data.py
    ├── test_models.py
    ├── test_trainers.py
    └── test_utils.py
```

### Folder Descriptions
- **scripts**: Command-line utilities for backtesting and evaluation.
- **agents**: Logic for creating, saving, and loading custom AI agents.
- **data**: Components handling data retrieval, preprocessing, and feature generation.
- **models**: Different neural architectures for time-series forecasting and signal generation.
- **trainers**: Training classes, each optimized for a specific model type.
- **utils**: Config files, logging, metrics, prediction, signal generation, and Twitter integration.
- **tests**: Basic unit tests to validate correctness and reliability.

## Advanced Topics

### Custom Feature Engineering
Enhance or replace `FeatureEngineering` with domain-specific transformations such as on-chain metrics or sentiment data integration.

### Hyperparameter Tuning
Tools like Optuna or Ray Tune can be integrated with the trainer classes to systematically optimize model hyperparameters.

### Distributed Training
For large-scale experiments, adapt the training logic (`trainer_lstm.py`, `trainer_transformer.py`, etc.) to run on multi-GPU or distributed systems via `torch.distributed`.

### Risk & Money Management
The codebase offers a technical signal generation pipeline, but position sizing, leverage, and risk controls are left to the end-user to develop further.

## Disclaimers

### Financial Risk
Cryptocurrency trading involves a high degree of risk. The signals generated by this project are for informational and educational purposes only and should not be construed as financial advice.

### No Warranty
All software is provided on an “AS IS” basis without warranties of any kind. The contributors disclaim liability for any financial losses or damages resulting from the use of this project.

### Regulatory Compliance
Users are responsible for abiding by local regulations regarding cryptocurrency trading, personal data storage, and social media automation.

## Contributing
We welcome enhancements, bug fixes, and feature additions via pull requests. If you encounter any issues or have ideas for improving the project, open an issue in the repository. Please ensure that any contributions maintain the existing architecture and adhere to standard best practices.

## License
This project is released under the MIT License. You are free to use, modify, and distribute this software in your own or commercial projects, subject to the license terms.

## Contact
For questions or detailed assistance:

We appreciate your interest in the Crypto Signals Project and hope it serves as a reliable foundation for your AI-driven trading strategies.
