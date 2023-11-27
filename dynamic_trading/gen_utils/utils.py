import multiprocessing as mp
import os
from typing import Union, Tuple, Any

import numpy as np
import pandas as pd
from joblib import load
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from numpy.polynomial import Polynomial
from scipy.stats import truncnorm

from dynamic_trading.enums.enums import InitialQvalueEstimateType, OptimizerType, SupervisedRegressorType

available_ann_architectures = [(64,),
                               (64, 32),
                               (64, 32, 8),
                               (64, 32, 16, 8),
                               (64, 32, 16, 8, 4)]


def read_ticker() -> str:
    """
    Reads security ticker from settings file.

    Returns
    -------
    ticker : str
        An ID to identify the traded security. If this ID is present in the list of available securities, the code will
        read its time series from the source data. Otherwise, it will try to download the time series from Yahoo
        finance via the :obj:`yfinance` module.

    """

    filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/resources/data/data_source/settings.csv'
    df_trad_params = pd.read_csv(filename, index_col=0)
    ticker = str(df_trad_params.loc['ticker', 'value'])

    return ticker


def get_available_futures_tickers() -> list:
    """
    Returns the pre-defined list of available tickers for the securities in
    :obj:`resources/data/data_source/market_data/`.

    Returns
    -------
    lst: list
        List of available securities tickers.

    """

    lst = ['cocoa', 'coffee', 'copper', 'WTI', 'WTI-spot', 'gold', 'lead', 'nat-gas-rngc1d', 'nat-gas-reuter',
           'nickel', 'silver', 'sugar', 'unleaded', 'zinc']  # 'tin', 'gasoil'

    return lst


def instantiate_polynomialFeatures(degree) -> PolynomialFeatures:
    """
    Instantiates a scikit-learn :class:`~sklearn.preprocessing.PolynomialFeatures` object for a given polynomial degree.
    By default, interactions are considered and bias is included.

    Parameters
    ----------
    degree : int
        The degree of the polynomial.

    Returns
    -------
    poly : :class:`~sklearn.preprocessing.PolynomialFeatures`
        :class:`~sklearn.preprocessing.PolynomialFeatures` object defining the features of a polynomial regression.

    """

    poly = PolynomialFeatures(degree=degree,
                              interaction_only=False,
                              include_bias=True)

    return poly


def find_polynomial_minimum(coef: Union[list, np.ndarray, tuple],
                            bounds: Union[list, np.ndarray, tuple]) -> Tuple[float, bool]:
    """
    Given a set of polynomial coefficients coef = :math:`(a_0, a_1, ..., a_n)`, it computes the point of minimum of the
    polynomial :math:`p(x) = a_0 + a_1  x + ... + a_n x^n` in a given interval ``[bounds[0], bounds[1]]``. If the
    minimum is not found, a random output is given as determined by the :obj:`scipy.stats.truncnorm` distribution
    function on the given interval and a flag is returned.

    Parameters
    ----------
    coef : array_like
        Polynomial coefficients, expressed in an array_like format.
    bounds : array_like
        Domain bounds, expressed in an array_like format.

    Returns
    -------
    x_optim : float
        Polynomial point of minimum.
    flag_error : bool
        Flag determining whether the minimum existed in the interval.

    """

    x_optim_when_error = truncnorm.rvs(a=bounds[0], b=bounds[1], loc=0., scale=0.01 * (bounds[1] - bounds[0]))
    flag_error = False

    if len(coef) < 2:
        raise NameError('Polynomial must be of degree >= 2')

    p = Polynomial(coef)
    dp = p.deriv(m=1)
    dp2 = p.deriv(m=2)

    stationary_points = dp.roots()

    # exclude complex roots
    stationary_points = np.real(stationary_points[np.isreal(stationary_points)])

    if len(stationary_points) == 0:
        x_optim = x_optim_when_error
        flag_error = True

    else:
        stationary_points = stationary_points[stationary_points >= bounds[0]]
        stationary_points = stationary_points[stationary_points <= bounds[1]]

        stationary_points_hessian = np.zeros(len(stationary_points))

        for i in range(len(stationary_points)):
            stationary_points_hessian[i] = dp2(stationary_points[i])

        minimal_points = stationary_points[stationary_points_hessian > 0]

        if len(minimal_points) > 0:
            x_optim = minimal_points[0]
            for i in range(len(minimal_points)):
                if p(minimal_points[i]) >= p(x_optim):
                    x_optim = minimal_points[i]
        else:
            x_optim = x_optim_when_error
            flag_error = True

    eps_plots = np.random.rand()
    if eps_plots < 10**-3:
        _make_plot_once_in_a_while(p, dp, dp2, bounds, x_optim, eps_plots)

    return x_optim, flag_error


def _make_plot_once_in_a_while(p, dp, dp2, bounds, x_optim, eps_plots):

    xx = np.linspace(bounds[0], bounds[1], 20)
    yy = p(xx)
    dyy = dp(xx)
    ddyy = dp2(xx)

    dpi = plt.rcParams['figure.dpi']
    plt.figure(figsize=(800 / dpi, 600 / dpi), dpi=dpi)
    plt.plot(xx, yy, label='Polinomial')
    plt.plot(xx, dyy, label='Polynomial 1st derivative')
    plt.plot(xx, ddyy, label='Polynomial 2nd derivative')
    plt.plot(xx, 0 * np.ones(len(xx)), label='Zero line', color='k')
    plt.vlines(x_optim, min(0, p(x_optim)), max(0, p(x_optim)), label=f'Optimum = {x_optim: .2f}')
    plt.legend()
    plt.xlim(bounds)

    filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filename += f'/resources/figures/polynomial/polynomial{int(eps_plots*10**5)}.png'

    plt.savefig(filename)


def read_trading_parameters_training() -> tuple[float, int, int, int, bool, int | None, InitialQvalueEstimateType, bool,
                                                bool, bool, bool, OptimizerType, SupervisedRegressorType, float, int,
                                                bool, int, int, str, float, bool, bool, int, bool, float, bool, Any]:
    # todo: move output description to README in settings.csv description
    """
    Service function that reads the trading parameters from the disk.

    Returns
    -------
    shares_scale : float
        Factor for rescaling the shares :math:`M`.
    j_episodes : int
        Number of episodes :math:`J`: to generate within each batch.
    n_batches : int
        Number of batches :math:`N_B`.
    t_ : int
        Length :math:`T` of each episode.
    parallel_computing_train : bool
        Boolean determining whether training is performed via parallel computing.
    n_cores : int
        Number of cores to use in parallel computing.
    initialQvalueEstimateType : :class:`~dynamic_trading.enums.enums.InitialQvalueEstimateType`
        Setting for the initialization of the state-action value function. Refer to
        :class:`~dynamic_trading.enums.enums.InitialQvalueEstimateType` for more details.
    predict_pnl_for_reward : bool
        Boolean determining whether the PnL is predicted in terms of the factor in the reward definition. If ``False``,
        then the reward is computed as
        :math:`R_{t+1} = \gamma(n^{'}_t x_{t+1} - 0.5\kappa n^{'}_t\Sigma n_t)-c(\Delta n_t)`; if ``True``, it is
        computed as
        :math:`R_{t+1} = \gamma(n^{'}_t g(f_t) - 0.5\kappa n^{'}_t\Sigma n_t)-c(\Delta n_t)` where
        :math:`g(f_t)=E[x_{t+1}|f_t]`.
    average_across_models : bool
        Boolean determining whether the SARSA algorithm performs model averaging across batches.
    use_best_n_batch : bool
        Boolean determining whether the last or the best available (average of) model should be used.
    train_benchmarking_GP_reward : bool
        Boolean determining whether the RL agent is being trained by benchmarking a GP agent. If this is true, then a
        AgentGP is instantiated; for each trade, both the RL and the GP rewards are computed. If the GP agent has
        outperformed the RL agent on the given trade, then the RL trade is substituted.
    optimizerType : :class:`~dynamic_trading.enums.enums.OptimizerType`
        Determines which global optimizer to use in the greedy policy optimization. Refer to
        :class:`~dynamic_trading.enums.enums.OptimizerType` for more details.
    supervisedRegressorType : :class:`~dynamic_trading.enums.enums.SupervisedRegressorType`
        Determines what kind of supervised regressor should be used to fit the state-action value function. Refer to
        :class:`~dynamic_trading.enums.enums.SupervisedRegressorType` for more details.
    eps_start : float
        Starting parameter :math:`\epsilon_0` for the :math:`\epsilon`-greedy policy.
    max_ann_depth : int
        Integer determining the depth of the Neural Network used to fit the state-action value function. It acts on
        pre-defined architectures given by [(64,), (64, 32), (64, 32, 8), (64, 32, 16, 8), (64, 32, 16, 8, 4)]
    early_stopping : bool
        Whether to use early stopping in the Neural Network fit. Refer to scikit-learn for more details.
    max_iter : int
        Maximum iteration in supervised regressor fit. Refer to scikit-learn for more details.
    n_iter_no_change : int
        Refer to scikit-learn for more details.
    activation : str
        Activation function used in Neural Network. Refer to scikit-learn for more details.
    alpha_sarsa : float
        Learning rate in SARSA updating formula.
    decrease_eps : bool
        Boolean determining whether :math:`\epsilon` should be decreased across batches.
    random_initial_state : bool
        Boolean determining whether the initial state :math:`s_0` is selected randomly.
    max_polynomial_regression_degree : int
        Maximum polynomial degree to be considered.
    max_complexity_no_gridsearch : bool
        Boolean determining whether the maximum Neural Network or Polynomial complexity should be used (``True``), or if
        a cross-validation should be performed (``False``). Refer to scikit-learn for more details on
        :class:`~sklearn.model_selection.GridSearchCV`.
    alpha_ewma : float
        Speed of the exponential weighting in the SARSA model averaging across batches.
    parallel_computing_sim : bool
        Boolean determining whether simulation testing is performed via parallel computing.
    use_best_n_batch_mode : str
        Determines the mode with which the "best batch" is selected. Can be any of the following: ``t_test_pvalue``,
        best choice is based on equality/outperforming criteria with respect to the benchmark based on the p-value of
        specific hypothesis tests; ``t_test_statistic``, best choice is based on equality/outperforming criteria with
        respect to the benchmark based on the statistic of specific hypothesis tests; ``reward``, best choice is based
        on the reward obtained in the training phase; ``average_q``, best choice is based on the average state-action
        value obtained in the training phase; ``model_convergence``, best choice is based on a convergence criterion on
        the norm of two subsequent state-action value function estimates; ``wealth_net_risk``, best choice is based on
        the wealth net risk obtained on simulated RL strategies at the end of each batch.

    """

    filename = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    filename += '/resources/data/data_source/settings.csv'
    df_trad_params = pd.read_csv(filename, index_col=0)

    shares_scale = float(load(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                              + '/resources/data/data_tmp/shares_scale.joblib'))

    j_episodes = int(df_trad_params.loc['j_episodes'][0])
    n_batches = int(df_trad_params.loc['n_batches'][0])
    t_ = int(df_trad_params.loc['t_'][0])

    if df_trad_params.loc['parallel_computing_train'][0] == 'Yes':
        parallel_computing_train = True
        n_cores = min(int(df_trad_params.loc['n_cores'][0]), mp.cpu_count())
    elif df_trad_params.loc['parallel_computing_train'][0] == 'No':
        parallel_computing_train = False
        n_cores = None
    else:
        raise NameError('Invalid value for parameter parallel_computing_train in settings.csv')

    if df_trad_params.loc['parallel_computing_sim'][0] == 'Yes':
        parallel_computing_sim = True
        n_cores = min(int(df_trad_params.loc['n_cores'][0]), mp.cpu_count())
    elif df_trad_params.loc['parallel_computing_sim'][0] == 'No':
        parallel_computing_sim = False
        n_cores = None
    else:
        raise NameError('Invalid value for parameter parallel_computing_sim in settings.csv')

    # if zero, the initial estimate of the qvalue function is 0; if random, it is N(0,1)
    initialQvalueEstimateType = InitialQvalueEstimateType(df_trad_params.loc['initialQvalueEstimateType'][0])

    # if True, the agent uses the model to predict the next step pnl and sig2 for the reward; else, uses the realized
    if df_trad_params.loc['predict_pnl_for_reward'][0] == 'Yes':
        predict_pnl_for_reward = True
    elif df_trad_params.loc['predict_pnl_for_reward'][0] == 'No':
        predict_pnl_for_reward = False
    else:
        raise NameError('Invalid value for parameter predict_pnl_for_reward in settings.csv')

    # if True, the agent averages across supervised regressors in its definition of q_value; else, uses the last one
    if df_trad_params.loc['average_across_models'][0] == 'Yes':
        average_across_models = True
    elif df_trad_params.loc['average_across_models'][0] == 'No':
        average_across_models = False
    else:
        raise NameError('Invalid value for parameter average_across_models in settings.csv')

    # if True, then the agent considers the supervised regressors only up to n<=n_batches, where n is the batch that
    # provided the best reward in the training phase
    if df_trad_params.loc['use_best_n_batch'][0] == 'Yes':
        use_best_n_batch = True
    elif df_trad_params.loc['use_best_n_batch'][0] == 'No':
        use_best_n_batch = False
    else:
        raise NameError('Invalid value for parameter use_best_n_batch in settings.csv')

    use_best_n_batch_mode = df_trad_params.loc['use_best_n_batch_mode'][0]

    # if True, the agent observes the reward GP would obtain and forces its strategy to be GP's if such reward is higher
    # than the one learned automatically
    if df_trad_params.loc['train_benchmarking_GP_reward'][0] == 'Yes':
        train_benchmarking_GP_reward = True
    elif df_trad_params.loc['train_benchmarking_GP_reward'][0] == 'No':
        train_benchmarking_GP_reward = False
    else:
        raise NameError('Invalid value for parameter train_benchmarking_GP_reward in settings.csv')

    # which optimizer to use in greedy policy
    optimizerType = OptimizerType(df_trad_params.loc['optimizerType'][0])

    # choose which model to use for supervised regression
    supervisedRegressorType = SupervisedRegressorType(df_trad_params.loc['supervisedRegressorType'][0])

    # initial epsilon for eps-greedy policy: at each batch iteration, we do eps <- eps/3
    eps_start = float(df_trad_params.loc['eps_start'][0])

    max_ann_depth = int(df_trad_params.loc['max_ann_depth'][0])

    if df_trad_params.loc['early_stopping'][0] == 'Yes':
        early_stopping = True
    elif df_trad_params.loc['early_stopping'][0] == 'No':
        early_stopping = False
    else:
        raise NameError('Invalid value for parameter early_stopping in settings.csv')

    max_iter = int(df_trad_params.loc['max_iter'][0])

    n_iter_no_change = int(df_trad_params.loc['n_iter_no_change'][0])

    activation = str(df_trad_params.loc['activation'][0])

    alpha_sarsa = float(df_trad_params.loc['alpha_sarsa'][0])

    if df_trad_params.loc['decrease_eps'][0] == 'Yes':
        decrease_eps = True
    elif df_trad_params.loc['decrease_eps'][0] == 'No':
        decrease_eps = False
    else:
        raise NameError('Invalid value for parameter decrease_eps in settings.csv')

    if df_trad_params.loc['random_initial_state'][0] == 'Yes':
        random_initial_state = True
    elif df_trad_params.loc['random_initial_state'][0] == 'No':
        random_initial_state = False
    else:
        raise NameError('Invalid value for parameter random_initial_state in settings.csv')

    max_polynomial_regression_degree = int(df_trad_params.loc['max_polynomial_regression_degree'][0])

    if df_trad_params.loc['max_complexity_no_gridsearch'][0] == 'Yes':
        max_complexity_no_gridsearch = True
    elif df_trad_params.loc['max_complexity_no_gridsearch'][0] == 'No':
        max_complexity_no_gridsearch = False
    else:
        raise NameError('Invalid value for parameter max_complexity_no_gridsearch in settings.csv')

    alpha_ewma = float(df_trad_params.loc['alpha_ewma'][0])

    return (shares_scale, j_episodes, n_batches, t_, parallel_computing_train, n_cores, initialQvalueEstimateType,
            predict_pnl_for_reward, average_across_models, use_best_n_batch, train_benchmarking_GP_reward,
            optimizerType, supervisedRegressorType, eps_start, max_ann_depth, early_stopping, max_iter,
            n_iter_no_change, activation, alpha_sarsa, decrease_eps, random_initial_state,
            max_polynomial_regression_degree, max_complexity_no_gridsearch, alpha_ewma, parallel_computing_sim,
            use_best_n_batch_mode)
