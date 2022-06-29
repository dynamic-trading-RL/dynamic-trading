# dynamic-trading
This project aims at using Reinforcement Learning to replicate and expand the model proposed in Gârleanu-Pedersen, Dynamic Trading with Predictable Returns and Transaction Costs.

This is a work-in-progress. For any questions, please refer to Federico Giorgi (fdr.giorgi@gmail.com).

Project requirements are listed in the requirements.txt file.

## Data
The folder dynamic-trading/data/data_source/market_data contains all the relevant data used to fit market dynamics, in particular
- futures_data.xlsx contains commodity futures time series taken into considerations
- SP500.xlsx contains the S&P 500 time series, which could be used as predicting factor

If the user provides a ticker that is not listed among the names of the commodity futures, the code will try to download that ticker from Yahoo Finance.

The folder dynamic-trading/data/data_source/settings contains the main settings that the code reads in order to fit the dynamics, train the agent and perform backtesting. Various options for the factor definition are available.

## Scripts
The main scripts are the following.
- s_calibrate_all_futures_market_dynamics.py: This script iterates on all commodity futures time series and fit all the possible models for risk drivers (Linear, NonLinear) and factors (AR, GARCH, TARCH, AR_TARCH). See more details in enums.py. This scripts plots the residuals of each factor dynamics and gives information on which factor dynamics is best for each specific future, as defined by the one which has minimum lag-one absolute residual absolute autocorrelation.
- s_calibrate_specific_market_dynamics.py: This script calibrates a specific asset as provided by the user (possibly, downloaded from Yahoo Finance, see previous section).
- s_train_agent.py: This script uses SARSA batch learning to train an agent to optimally trade the selected asset. Various settings are available in the above mentioned folder, such as number of batches, number of episodes, length of each episode, whether to use parallel computing or not etc.
- s_backtesting.py: Compares the RL agent against the benchmark agents provided by Markowitz and Gârleanu-Pedersen. If the RL is trained on models that are compatible with the setting of Gârleanu-Pedersen (AR(1) model on the factor and linear model on the P&L) then the agent is expected to replicate Gârleanu-Pedersen. If the agent is trained on alternative models that best capture the true market dynamics, then RL should outperform Gârleanu-Pedersen.
