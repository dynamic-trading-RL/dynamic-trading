# dynamic-trading
This project aims at using Reinforcement Learning to replicate and expand the model proposed in Gârleanu-Pedersen, Dynamic Trading with Predictable Returns and Transaction Costs.

This is a work-in-progress. For any questions, please refer to Federico Giorgi (fdr.giorgi@gmail.com).

Project requirements are listed in the requirements.txt file.

## Data
The folder dynamic-trading/data/data_source/market_data contains all the relevant data used to fit market dynamics, in particular
- assets_data.xlsx contains commodity futures time series taken into considerations
- SP500.xlsx contains the S&P 500 time series, which could be used as predicting factor
- VIX.xlsx contains the S&P 500 time series, which could be used as predicting factor
- RV5.xlsx contains the S&p 500 5-minutes realized variance, which could be used as predicting factor

If the user provides a ticker that is not listed among the names of the commodity futures, the code will try to download that ticker from Yahoo Finance.

The folder dynamic-trading/data/data_source/settings contains the main settings that the code reads in order to fit the dynamics, train the agent and perform backtesting. Various options for the factor definition are available. In particular:
- ticker: specifies the ticker of the asset to be taken into consideration (e.g. 'WTI')
- riskDriverDynamicsType: the dynamics for the variable 'x', can be 'Linear' or 'NonLinear' in the factor
- riskDriverType: the nature of the variable 'x', can be 'PnL' or 'Return'
- factor_ticker: the ticker of the factor; if not provided, then the factor is constructed starting from the asset, otherwise, it is loaded from the available data (e.g. 'VIX' or 'SP500')
- factorComputationType: determines whether the factor is computed as a 'MovingAverage' or a 'StdMovingAverage'
- window: the window for the moving average; if = 1, then no moving average transformation is applied
- factorTransformationType: can be 'Diff' or 'LogDiff', and it determines whether the code needs to take the level or the log-level of the input series (e.g. the realized variance or the log-realized variance)
- factorDynamicsType: can be 'AR', 'SETAR', 'GARCH', 'TARCH' or 'AR_TARCH'
- factor_in_state: 'Yes' or 'No', determines whether the factor should be in the state variable
- ttm_in_state: 'Yes' or 'No', determines whether the time to maturity should be in the state variable
- price_in_state: 'Yes' or 'No', determines whether the price should be in the state variable
- pnl_in_state: 'Yes' or 'No', determines whether the pnl should be in the state variable
- GP_action_in_state: 'Yes' or 'No', determines whether the GP action should be in the state variable
- strategyType: 'Unconstrained' or 'LongOnly', determines the strategy of the agent
- randomActionType: can be 'RandomUniform' (value function is initialized randomly and uniformly), 'RandomTruncNorm' (value function is initialized randomly and truncated normally) or 'GP' (agent follows GP if value function is not given)
- gamma: discount factor in RL target
- start_date: time series will start from here
- in_sample_proportion: the proportion of the complete time series on which calibrating the dynamics (and hence the agent)
- j_episodes: number of episodes within each batch
- t_: length of each episode
- n_batches: number of batches
- parallel_computing: can be 'Yes' (uses parallel computing) or 'No' (do not use parallel computing)
- n_cores: number of cores in case parallel_computing = 'Yes'

## Scripts
The main scripts are the following.

### s_calibrate_all_futures_market_dynamics.py
This script iterates on all commodity futures time series and fit all the possible models for risk drivers (Linear, NonLinear) and factors (AR, GARCH, TARCH, AR_TARCH). See more details in enums.py. This scripts plots the residuals of each factor dynamics and gives information on which factor dynamics is best for each specific future, as defined by the one which has minimum lag-one absolute residual absolute autocorrelation.

### s_calibrate_specific_market_dynamics.py
This script calibrates a specific asset as provided by the user (possibly, downloaded from Yahoo Finance, see previous section).

### s_train_agent.py
This script uses SARSA batch learning to train an agent to optimally trade the selected asset. Various settings are available in the above mentioned folder, such as number of batches, number of episodes, length of each episode, whether to use parallel computing or not etc.

### s_backtesting.py
Compares the RL agent against the benchmark agents provided by Markowitz and Gârleanu-Pedersen. If the RL is trained on models that are compatible with the setting of Gârleanu-Pedersen (AR(1) model on the factor and linear model on the P&L) then the agent is expected to replicate Gârleanu-Pedersen. If the agent is trained on alternative models that best capture the true market dynamics, then RL should outperform Gârleanu-Pedersen.
