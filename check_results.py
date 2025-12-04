import numpy as np
import pandas as pd
import os

try:
    mcmc = np.load('results/models/mcmc_rolling_results.npz')
    vol = mcmc['rolling_vol']
    print(f"MCMC Vol Mean: {np.nanmean(vol):.4f}")
    print(f"MCMC Vol Max: {np.nanmax(vol):.4f}")
    
    garch = pd.read_csv('results/models/garch_results.csv')
    print(f"GARCH Vol Mean: {garch['Volatility'].mean():.4f}")
except Exception as e:
    print(e)
