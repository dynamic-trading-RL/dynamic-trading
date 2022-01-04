# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 14:51:24 2021

@author: feder
"""

import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
from dt_functions import (ReturnDynamics, FactorDynamics,
                          ReturnDynamicsType, FactorDynamicsType,
                          MarketDynamics, Market,
                          read_return_parameters,
                          read_factor_parameters)

# Input parameters
j_ = 10
t_ = 50

calibration_parameters = pd.read_excel('data/calibration_parameters.xlsx',
                                       index_col=0)
ticker = calibration_parameters.loc['ticker', 'calibration-parameters']
startPrice = calibration_parameters.loc['startPrice', 'calibration-parameters']

return_is_pnl = load('data/return_is_pnl.joblib')

# Specify model
for returnDynamicsType in ReturnDynamicsType:
    for factorDynamicsType in FactorDynamicsType:

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
        market = Market(marketDynamics, startPrice, return_is_pnl=return_is_pnl)

        # Simulations
        market.simulate(j_=j_, t_=t_)

        # Plots
        f = market._simulations['f']
        r = market._simulations['r']

        fig = plt.figure()
        for j in range(min(50, j_)):
            plt.plot(r[j, :], color='k', alpha=0.3)
        plt.xlabel('t')
        plt.ylabel('r')
        s = 'Simulated return; ' + factorDynamicsType.value + ' and ' +\
            returnDynamicsType.value
        plt.title(s)
        plt.savefig('figures/' + ticker + '-' + s + '.png')

        fig = plt.figure()
        for j in range(min(50, j_)):
            plt.plot(np.cumsum(r[j, :]), color='b', alpha=0.3)
        plt.xlabel('t')
        plt.ylabel('r')
        s = 'Simulated cumulative return; ' + factorDynamicsType.value +\
            ' and ' + returnDynamicsType.value
        plt.title(s)
        plt.savefig('figures/' + ticker + '-' + s + '.png')

        fig = plt.figure()
        for j in range(min(50, j_)):
            plt.plot(f[j, :], color='r', alpha=0.3)
        plt.xlabel('t')
        plt.ylabel('f')
        s = 'Simulated factor; ' + factorDynamicsType.value
        plt.title(s)
        plt.savefig('figures/' + ticker + '-' + s + '.png')

        fig = plt.figure()
        plt.scatter(f[:, :-1].flatten(), r[:, 1:].flatten(), s=2)
        plt.xlabel('f')
        plt.ylabel('r')
        s = 'Simulated factor vs return; ' + factorDynamicsType.value +\
            ' vs ' + returnDynamicsType.value
        plt.title(s)
        plt.savefig('figures/' + ticker + '-' + s + '.png')
