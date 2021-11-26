# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 12:41:52 2021

@author: Giorgi
"""

import pandas as pd
from dt_functions import (ReturnDynamics, FactorDynamics,
                          ReturnDynamicsType, FactorDynamicsType,
                          MarketDynamics, Market,
                          read_return_parameters,
                          read_factor_parameters,
                          simulate_market,
                          generate_episode)

# Input parameters
j_episodes = 100
n_batches = 5
t_ = 50

calibration_parameters = pd.read_excel('data/calibration_parameters.xlsx',
                                       index_col=0)
ticker = calibration_parameters.loc['ticker', 'calibration-parameters']

returnDynamicsType = ReturnDynamicsType.Linear
factorDynamicsType = FactorDynamicsType.AR

# Instantiate dynamics
returnDynamics = ReturnDynamics(returnDynamicsType)
factorDynamics = FactorDynamics(factorDynamicsType)

# Read calibrated parameters
return_parameters = read_return_parameters(returnDynamicsType)
factor_parameters = read_factor_parameters(factorDynamicsType)

# Set dynamics
returnDynamics.set_parameters(return_parameters)
factorDynamics.set_parameters(factor_parameters)
marketDynamics = MarketDynamics(returnDynamics=returnDynamics,
                                factorDynamics=factorDynamics)
market = Market(marketDynamics)

# Simulations
r, f = simulate_market(market, j_episodes, n_batches, t_)

generate_episode(# dummy for parallel computing
                 j,
                 # market parameters
                 Lambda, mu, Sigma,
                 # market simulations
                 f,
                 # RL parameters
                 eps, rho, q_value, alpha, gamma, lot_size,
                 optimizers,
                 optimizer)
