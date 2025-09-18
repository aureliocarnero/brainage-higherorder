import glob
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr


files = glob.glob('OASIS3/realizations/run*.txt')

estimates = []
for f in files:
    pred = np.loadtxt(f,unpack=True,usecols=([2]))
    estimates.append(pred)

means = np.mean(estimates, axis=0)
stds = np.std(estimates, axis=0)

true = np.loadtxt(f,unpack=True,usecols=([1]))
mrid = np.loadtxt(f,unpack=True,usecols=([0]), dtype=str)
print('PEARSON', pearsonr(true,means))
print('Mean Absolute error', mean_absolute_error(true, means))



