# AI Trading System

## Overview

A comprehensive AI-powered trading system implementing Smart Money Concepts (SMC) and Inner Circle Trader (ICT) strategies for the Indian stock market. This system combines advanced machine learning models with sophisticated feature engineering to create an autonomous trading agent that identifies institutional market patterns and executes trades with built-in risk management.

## 🎯 Key Features

- **Smart Money Concepts (SMC) & ICT Pattern Recognition**: Automated detection of Order Blocks, Fair Value Gaps, Market Structure Shifts, and Liquidity Grabs
- **Hybrid AI Architecture**: Combines LSTM-based prediction models with Reinforcement Learning agents for optimal decision-making
- **Event-Driven System Architecture**: Scalable, modular design with real-time data processing
- **SEBI Compliance**: Full compliance with Indian algorithmic trading regulations
- **Advanced Risk Management**: Multi-layer risk controls with automated kill switches
- **Real-time Market Data Integration**: Support for NSE/BSE data feeds
- **Comprehensive Backtesting Framework**: Rigorous historical testing with proper cost modeling

## 🏗️ System Architecture

The system follows an event-driven architecture with the following core components:

### Core Modules

1. **Data Handler** - Real-time and historical market data ingestion
2. **Feature Engine** - ICT/SMC pattern detection and technical indicator computation
3. **AI Decision Engine** - LSTM + Reinforcement Learning hybrid model
4. **Risk Management Module** - Pre-trade risk controls and position sizing
5. **Execution Handler** - Broker API integration with SEBI compliance
6. **Portfolio Manager** - Real-time portfolio tracking and P&L management
7. **Monitoring Dashboard** - System oversight and audit trail logging

## 📊 Feature Engineering Pipeline

### Market Structure Analysis
- Swing High/Low detection with multiple timeframe analysis
- Break of Structure (BOS) and Change of Character (CHoCH) identification
- Premium/Discount zone mapping using Fibonacci levels

### ICT/SMC Pattern Detection
- **Order Blocks**: Last opposing candle before structure breaks
- **Fair Value Gaps**: Three-candle imbalance patterns
- **Liquidity Grabs**: Stop hunt detection with volume confirmation
- **Equal Highs/Lows**: Liquidity pool identification

### Technical Indicators
- Trend indicators (SMA, EMA, MACD)
- Momentum oscillators (RSI, Stochastic)
- Volatility measures (Bollinger Bands, ATR)
- Volume analysis (OBV, Volume Profile)

### Alternative Data Integration
- Financial news sentiment analysis using FinBERT/LLM models
- Real-time sentiment scoring and feature integration

## 🤖 AI/ML Architecture

### Supervised Learning Component (LSTM)
```python
# LSTM Model for Pattern Recognition
- Input: Multi-dimensional feature vectors (ICT/SMC + Technical + Sentiment)
- Architecture: Stacked LSTM layers with dropout regularization
- Output: Probability scores for market direction prediction
```

### Reinforcement Learning Component (PPO)
```python
# RL Agent for Trading Policy Optimization
- State: Market features + Portfolio state + LSTM predictions
- Action Space: {HOLD, GO_LONG, GO_SHORT}
- Reward Function: Risk-adjusted returns (Sharpe ratio optimization)
- Algorithm: Proximal Policy Optimization (PPO)
```

### Hybrid Integration
The system uses a two-stage approach:
1. LSTM generates probabilistic market forecasts
2. RL agent uses these predictions + market state to optimize trading actions

## 📋 Project Structure

```
ai-trading-system/
├── src/
│   ├── data/
│   │   ├── handlers/          # Data ingestion and processing
│   │   ├── providers/         # Market data provider integrations
│   │   └── storage/           # Time-series database management
│   ├── features/
│   │   ├── ict_smc/          # ICT/SMC pattern detection
│   │   ├── technical/         # Technical indicator computation
│   │   ├── sentiment/         # NLP sentiment analysis
│   │   └── engineering/       # Feature pipeline orchestration
│   ├── models/
│   │   ├── supervised/        # LSTM implementation
│   │   ├── reinforcement/     # RL agent implementation
│   │   └── hybrid/           # Integrated model architecture
│   ├── execution/
│   │   ├── brokers/          # Broker API integrations
│   │   ├── orders/           # Order management system
│   │   └── compliance/       # SEBI regulation compliance
│   ├── risk/
│   │   ├── controls/         # Pre-trade risk checks
│   │   ├── portfolio/        # Position and portfolio management
│   │   └── monitoring/       # Real-time risk monitoring
│   ├── backtesting/
│   │   ├── engine/           # Backtesting framework
│   │   ├── metrics/          # Performance evaluation
│   │   └── optimization/     # Strategy parameter tuning
│   └── infrastructure/
│       ├── events/           # Event-driven messaging
│       ├── logging/          # Comprehensive audit trails
│       └── monitoring/       # System health monitoring
├── config/
│   ├── trading/              # Trading strategy configurations
│   ├── models/               # ML model hyperparameters
│   ├── brokers/              # Broker API configurations
│   └── compliance/           # SEBI compliance settings
├── data/
│   ├── raw/                  # Raw market data storage
│   ├── processed/            # Cleaned and featured data
│   └── models/               # Trained model artifacts
├── notebooks/
│   ├── research/             # Strategy research and development
│   ├── backtesting/          # Backtesting analysis
│   └── monitoring/           # Performance analysis
├── tests/
│   ├── unit/                 # Unit tests for all modules
│   ├── integration/          # Integration testing
│   └── compliance/           # SEBI compliance testing
└── docs/
    ├── architecture/         # System design documentation
    ├── compliance/           # Regulatory compliance guide
    └── deployment/           # Production deployment guide
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for ML training)
- Static IP address (required for SEBI compliance)
- Indian stock broker account with API access
- Market data subscription (NSE/BSE authorized vendor)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ajaygm18/ai-trading-system.git
cd ai-trading-system
```

2. **Set up Python environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**:
```bash
cp .env.example .env
# Edit .env with your broker API credentials and data feed settings
```

4. **Initialize the database**:
```bash
python scripts/setup_database.py
```

5. **Download initial market data**:
```bash
python scripts/download_historical_data.py --symbols NIFTY50 --period 5Y
```

### Quick Start

1. **Run backtesting**:
```bash
python -m src.backtesting.runner --config config/strategies/ict_smc_base.yaml
```

2. **Train ML models**:
```bash
python -m src.models.train --model lstm --config config/models/lstm_base.yaml
python -m src.models.train --model rl --config config/models/rl_ppo.yaml
```

3. **Start paper trading**:
```bash
python -m src.main --mode paper --config config/trading/live_paper.yaml
```

4. **Launch monitoring dashboard**:
```bash
streamlit run src/monitoring/dashboard.py
```

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Total return, annualized return, risk-adjusted returns
- **Risk Metrics**: Maximum drawdown, Sharpe ratio, Sortino ratio, Calmar ratio
- **Trade Statistics**: Win rate, profit factor, average trade duration
- **Execution Metrics**: Slippage, transaction costs, order fill rates
- **Model Performance**: Prediction accuracy, feature importance, model drift

## 🔒 SEBI Compliance Features

- **Algorithm Registration**: Automated submission workflows for broker approval
- **Unique Algo ID**: Automatic tagging of all orders with SEBI-mandated identifiers
- **Audit Trails**: Comprehensive logging of all system decisions and trades
- **Risk Controls**: Hard-coded position limits and kill switches
- **Secure API Access**: OAuth 2.0 + 2FA implementation with IP whitelisting
- **Order Frequency Monitoring**: Automatic throttling to stay within regulatory limits

## 🔧 Configuration

### Trading Strategy Configuration
```yaml
# config/strategies/ict_smc_base.yaml
strategy:
  name: "ICT_SMC_Hybrid"
  timeframe: "1D"
  universe: "NIFTY100"
  
risk_management:
  max_position_size: 0.05  # 5% of portfolio per position
  max_daily_loss: -0.02    # 2% daily stop loss
  max_drawdown: -0.10      # 10% maximum drawdown
  
features:
  ict_smc:
    enable_order_blocks: true
    enable_fair_value_gaps: true
    enable_liquidity_grabs: true
    swing_detection_periods: [5, 10, 20]
  
  technical_indicators:
    enable_trend: true
    enable_momentum: true
    enable_volatility: true
    enable_volume: true
```

### Model Configuration
```yaml
# config/models/lstm_base.yaml
model:
  type: "LSTM"
  sequence_length: 60
  layers:
    - type: "LSTM"
      units: 50
      return_sequences: true
      dropout: 0.2
    - type: "LSTM"
      units: 50
      dropout: 0.2
    - type: "Dense"
      units: 1
      activation: "sigmoid"
  
training:
  batch_size: 32
  epochs: 100
  validation_split: 0.2
  early_stopping: true
```

## 📈 Backtesting Results

*Note: Results will be updated after initial backtesting runs*

### Strategy Performance Summary
- **Total Return**: TBD
- **Sharpe Ratio**: TBD
- **Maximum Drawdown**: TBD
- **Win Rate**: TBD
- **Profit Factor**: TBD

### Model Performance
- **LSTM Prediction Accuracy**: TBD
- **RL Agent Cumulative Reward**: TBD
- **Feature Importance Analysis**: TBD

## 🛠️ Development Roadmap

### Phase 1: Foundation (Months 1-2)
- [x] Repository setup and project structure
- [ ] Core data handling infrastructure
- [ ] Basic ICT/SMC pattern detection
- [ ] Initial LSTM model implementation

### Phase 2: Core Features (Months 3-4)
- [ ] Complete feature engineering pipeline
- [ ] Reinforcement learning agent development
- [ ] Backtesting framework implementation
- [ ] Risk management module

### Phase 3: Integration (Months 5-6)
- [ ] Broker API integration
- [ ] Real-time data feed connections
- [ ] Event-driven architecture implementation
- [ ] SEBI compliance features

### Phase 4: Testing & Optimization (Months 7-8)
- [ ] Comprehensive backtesting
- [ ] Paper trading implementation
- [ ] Performance optimization
- [ ] Security and compliance auditing

### Phase 5: Production Deployment (Months 9-10)
- [ ] Live trading implementation
- [ ] Monitoring and alerting systems
- [ ] Documentation and user guides
- [ ] Continuous improvement framework

## 🤝 Contributing

This is currently a personal project. If you're interested in contributing or have suggestions, please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important Risk Warning**: This trading system is for educational and research purposes. Trading in financial markets involves substantial risk and may result in complete loss of invested capital. Past performance does not guarantee future results. Always conduct thorough testing and consider consulting with financial professionals before using any automated trading system with real money.

**Regulatory Compliance**: Ensure compliance with all applicable regulations including SEBI guidelines for algorithmic trading in India. The user is responsible for obtaining necessary approvals and maintaining compliance with all regulatory requirements.

## 📞 Support

For questions, issues, or discussions:
- Create an issue in this repository
- Check the [documentation](docs/) for detailed guides
- Review the [compliance guide](docs/compliance/) for regulatory requirements

## 🙏 Acknowledgments

- SEBI for providing clear algorithmic trading guidelines
- ICT/SMC community for pattern identification methodologies
- Open source ML/AI libraries (TensorFlow, PyTorch, Scikit-learn)
- Financial data providers supporting Indian markets
- Python trading community for frameworks and tools

---

**Built with ❤️ for the Indian algorithmic trading community**