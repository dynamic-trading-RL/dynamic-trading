# dynamic-trading
This project aims at using Reinforcement Learning to replicate and expand the model proposed in Gârleanu-Pedersen, Dynamic Trading with Predictable Returns and Transaction Costs.

This is a work-in-progress. For any questions, please refer to Federico Giorgi (fdr.giorgi@gmail.com).

Project requirements are listed in the requirements.txt file.

## Data
The folder /resources/data/data_source contains all the relevant data used to fit market dynamics.
Particularly relevant are
- assets_data.xlsx contains commodity futures time series taken into considerations
- SP500.xlsx contains the S&P 500 time series, which could be used as predicting factor
- VIX.xlsx contains the S&P 500 time series, which could be used as predicting factor
- RV5.xlsx contains the S&p 500 5-minutes realized variance, which could be used as predicting factor
- commodities-summary-statistics.xlsx contains some information about the commodity futures, such as exchanged volumes

The user can integrate the file assets_data.xlsx and add additional factor files. If the user provides a ticker that is not listed among the names of the commodity futures, the code will try to download that ticker from Yahoo Finance.

The file /resources/data/data_source/settings.csv contains the main settings that the code reads in order to fit the dynamics, train the agent and perform testing. Various options for the factor definition are available. In particular:
- activation: the activation function to be used in case a Neural Network is fitted to the state-action value function. Refer to scikit-learn for more information.
- alpha_ewma: the parameter for performing exponential weighting when doing model averaging of models fitted on consecutive batches during SARSA learning.
- alpha_sarsa: the learning rate in the SARSA updating formula.
- average_across_models: 'Yes' or 'No', determines whether the code performs averaging on models fitted on consecutive batches. If 'No', the last fitted model is considered at the next batch.
- use_quadratic_cost_in_markowitz: 'Yes' or 'No', determines whether the Markowitz strategy considers the quadratic transaction costs in its formulation.
- decrease_eps: 'Yes' or 'No', determines whether the epsilon parameter in the epsilon-greedy strategy is decreased along batches.
- early_stopping: 'Yes' or 'No', determines whether to use early stopping to terminate training when validation score is not improving in the supervised regressor fitting.
- end_date: optional last date for time series considered by the code.
- eps_start: initial value for epsilon-greedy strategy.
- factorComputationType: determines whether the factor is computed as a 'MovingAverage' or a 'StdMovingAverage'; in the last case, it is a moving average divided by the moving standard deviation.
- factorDynamicsType: can be 'AR', 'SETAR', 'GARCH', 'TARCH' or 'AR_TARCH' and it determines the factor dynamics.
- factorTransformationType: can be 'Diff' or 'LogDiff' and it determines whether the code needs to take the level or the log-level of the input series (e.g. the realized variance or the log-realized variance).
- factor_ticker: the ticker of the factor; if not provided, then the factor is constructed starting from the asset, otherwise, it is loaded from the available data (e.g. 'VIX' or 'SP500').
- gamma: discount factor for cumulative reward in RL target.
- in_sample_proportion: the proportion of the complete time series on which calibrating the dynamics (and hence the agent).
- initialQvalueEstimateType: can be 'random' or 'zero' and it determines the initialization given to the state-action value function estimate.
- j_episodes: number of episodes within each batch.
- j_oos: number of episodes in out-of-sample testing.
- max_ann_depth: maximum depth for the neural network; it acts on pre-defined architectures.
- max_complexity_no_gridsearch: 'Yes' or 'No', determines whether the maximum supervised regressor is considered; if 'No', the code will execute tune the supervised regressor hyperparameters via cross-validation.
- max_iter: maximum number of iterations in supervised regressor fit.
- max_polynomial_regression_degree: maximum degree for polynomial regression.
- n_batches: number of batches.
- n_cores: number of cores in case parallel_computing = 'Yes'.
- n_iter_no_change: Maximum number of epochs to not meet improvement. Refer to scikit-learn documentation.
- optimizerType: global optimizer used to define the greedy policy.
- parallel_computing_train: can be 'Yes' (uses parallel computing) or 'No' (do not use parallel computing); specific for the training phase.
- parallel_computing_sim: can be 'Yes' (uses parallel computing) or 'No' (do not use parallel computing); specific for the out-of-sample simulation phase.
- predict_pnl_for_reward: 'Yes' or 'No', determines whether we use the expected-reward predictive formula.
- randomActionType: can be 'RandomUniform' (value function is initialized randomly and uniformly), 'RandomTruncNorm' (value function is initialized randomly and truncated normally) or 'GP' (agent follows GP if value function is not given)
- random_initial_state: 'Yes' or 'No', determines whether the initial state in each episode is fixed or randomly generated.
- riskDriverDynamicsType: the dynamics for the variable 'x', can be 'Linear' or 'NonLinear' in the factor.
- riskDriverType: the nature of the variable 'x', can be 'PnL' or 'Return'.
- start_date: optional first date for time series considered by the code
- state_average_past_pnl: 'Yes' or 'No', determines whether to include an average of the last security P\&Ls in the state.
- state_factor: 'Yes' or 'No', determines whether the factor should be in the state variable.
- state_GP_action: 'Yes' or 'No', determines whether the GP action should be in the state variable.
- state_pnl: 'Yes' or 'No', determines whether the pnl should be in the state variable.
- state_price: 'Yes' or 'No', determines whether the price should be in the state variable.
- state_ttm: 'Yes' or 'No', determines whether the time to maturity should be in the state variable.
- strategyType: 'Unconstrained' or 'LongOnly', determines the strategy of the agent.
- supervisedRegressorType: determines which supervised regressor to use for state-action value function fitting.
- t_: length of each episode
- t_test_mode: determines on which samples the t-tests are executed
- ticker: specifies the ticker of the asset to be taken into consideration (e.g. 'WTI')
- train_benchmarking_GP_reward: 'Yes' or 'No', determines whether the RL agent should use GP as a benchmark in the training phase.
- use_best_batch: 'Yes' or 'No', determines whether the code selects a particular batch as 'best' for output trained agent.
- use_best_n_batch_mode: determines the criterion for choosing the best batch. Can be either of the following
  - 't_test_pvalue': a hypothesis testing is performed at the end of each batch
    - if the specific experiment is trying to replicate the GP benchmark, then the hypothesis is {H0: final RL wealth = final GP wealth} and the best batch will be the one providing the largest p-value.
    - if the specific experiment is trying to outperform the GP benchmark, then the hypothesis is {H0: final RL wealth <= final GP wealth} and the best batch will be the one providing the smallest p-value.
  - 't_test_statistic': a hypothesis testing is performed at the end of each batch, similar to the previous case; the best batch is selected as that with the largest statistics (implying that RL is greater than GP).
  - 'reward': the best batch is the one that generated the best reward along its episodes.
  - 'average_q': the best batch is the one that generated the best average value function along its episodes.
  - 'model_convergence': the best batch is the one for which relative distance of value function is smallest relative to the previous batch.
  - 'wealth_net_risk': at the end of the batch, after fitting the value function, many episodes are generated out-of-sample; the best batch is the one that generated the best out-of-sample wealth_net_risk (average/std).
- window: the window for the moving average; if = 1, then no moving average transformation is applied

## Scripts
The main scripts are the following.

### s_calibrate_all_futures_market_dynamics.py
This script iterates on all commodity futures time series and fit all the possible models for risk drivers (Linear, NonLinear) and factors (AR, GARCH, TARCH, AR_TARCH). See more details in enums.py. This scripts plots the residuals of each factor dynamics and gives information on which factor dynamics is best for each specific future, as defined by the one which has minimum lag-one absolute residual absolute autocorrelation.

### s_calibrate_specific_market_dynamics.py
This script calibrates a specific asset as provided by the user (possibly, downloaded from Yahoo Finance, see previous section).

### s_compute_shares_scale.py
This script starts from a calibration and generates many episodes where the Markowitz strategy is applied. Then, it computes a quantile on the absolute shares possessed by the Markowitz agent along its simulations and sets this as bound for the RL shares, under the assumption that the magnitude of the Markowitz strategy is greater than that of the RL strategy, since it does not consider transaction costs.

### s_train_agent.py
This script uses SARSA batch learning to train an agent to optimally trade the selected asset. Various settings are available in the above-mentioned folder, such as number of batches, number of episodes, length of each episode, whether to use parallel computing or not etc.

### s_backtesting.py
Compares the RL agent against the benchmark agents provided by Markowitz and Gârleanu-Pedersen on realized market realizations of the security being considered. If the RL is trained on models that are compatible with the setting of Gârleanu-Pedersen (AR(1) model on the factor and linear model on the P&L) then the agent is expected to replicate Gârleanu-Pedersen. If the agent is trained on alternative models that best capture the true market dynamics, then RL should outperform Gârleanu-Pedersen.

### s_simulationtesting.py
Compares the RL agent against the benchmark agents provided by Markowitz and Gârleanu-Pedersen on simulated market realizations of the security being considered. If the RL is trained on models that are compatible with the setting of Gârleanu-Pedersen (AR(1) model on the factor and linear model on the P&L) then the agent is expected to replicate Gârleanu-Pedersen. If the agent is trained on alternative models that best capture the true market dynamics, then RL should outperform Gârleanu-Pedersen. The performance is evaluated via t tests.
