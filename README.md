# Portfolio Analysis with Bitcoin

This repository contains code and data for analyzing optimal Bitcoin allocation in retail investor portfolios.

## Repository Structure

### Code
- `portfolio_analysis_test.m`: Main MATLAB script for portfolio optimization and analysis

### Data
- `bitcoin_data.csv`: Bitcoin historical price data
- `bonds_data.csv`: Bond market historical price data
- `sp500_data.csv`: S&P 500 historical price data
- `TB3MS.csv`: 3-Month Treasury Bill rates (risk-free rate)

### Outputs
#### Figures
- `correlation_heatmap.pdf`: Asset correlation visualization
- `garch_volatility_labeled.pdf`: Bitcoin volatility analysis with GARCH model
- `efficient_frontier_onepanel_clean.pdf`: Efficient frontier visualization
- `bitcoin_allocation_by_risk_profile.pdf`: Bitcoin allocation across risk profiles
- `portfolio_weights.pdf`: Portfolio composition visualization
- `portfolio_value_over_time.pdf`: Historical performance analysis
- `risk_return_bitcoin_allocation.pdf`: Risk-return profile visualization
- Monte Carlo simulation results:
  - `monte_carlo_low_risk.pdf`
  - `monte_carlo_moderate_risk.pdf`
  - `monte_carlo_high_risk.pdf`

#### Results
- `portfolio_performance_metrics.csv`: Comprehensive performance metrics
- `portfolio_summary_metrics.csv`: Summary of portfolio statistics

## Analysis Results
The analysis examines three portfolio profiles:
1. Low-Risk: 1.0% Bitcoin, 39.0% S&P 500, 60.0% Bonds
2. Moderate-Risk: 5.0% Bitcoin, 55.0% S&P 500, 40.0% Bonds
3. High-Risk: 20.0% Bitcoin, 80.0% S&P 500, 0.0% Bonds

Each profile is optimized for maximum Sharpe ratio within its risk constraints. 