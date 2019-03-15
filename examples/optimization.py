import numpy as np
import pandas as pd

from sgoop.utilities import read_plumed_file, reweight_ct
from sgoop.sgoop import optimize_rc
from sgoop.visualization import plot_pmf

import os.path as op

data_path = '/Users/joshsmith/Work/uremic_toxins/IS_sgoop_results'

# Specify the filenames for your biased and unbiased runs
max_cal_file = op.join(data_path, 'max_cal/COLVAR_10ns')  # unbiased colvar file
# specify collective variables and bias column (if applicable)
cv_cols = [
    'L383_dist', 'N387_dist', 'R406_dist', 'Y407_dist',
    'K410_dist', 'L426_dist', 'V429_dist', 'L449_dist',
    'S485_dist', 'h1_h2', 'h1_h3', 'h2_h3'
]

rbias_col = 'metad.rbias'
# load MaxCal dataframe from PLUMED colvar file
max_cal_traj = read_plumed_file(max_cal_file, cv_columns=cv_cols)
max_cal_traj = max_cal_traj.iloc[:, :]

metad_trajectories = []
weights = []

unbinding_cvs = ['N387', 'Y407', 'L426', 'V429', 'L449']  # , 'K410']
for cv in unbinding_cvs:
    # load metad CV dataframe and rbias series (for c(t) reweighting
    metad_file = op.join(data_path, f'trial_rc/{cv}/COLVAR')  # biased colvar file
    metad_traj, rbias = read_plumed_file(
        metad_file, cv_columns=cv_cols, bias_column=rbias_col
    )
    # calculate frame weights from rbias
    metad_trajectories.append(metad_traj)
    weights.append(reweight_ct(rbias, kt=2.5))

# combine samples from all unbinding trajectories
# take every tenth entry from MetaD to reduce computational cost
metad_traj = pd.concat(metad_trajectories, ignore_index=True)[::10]
weights = np.concatenate(weights)[::10]

# specify columns you want to require
sgoop_params = {
    # adjust sgoop params
    'rc_bins': 50,
    # 'kde_bw': 0.05,
    'd': 3,
    'wells': 2,
    'diffusivity': None,
}


# assign weight to biased CV from trial run
# np.random.seed(24)
x0 = np.array([0 for _ in range(12)])
x0[3] = 1
# x0 = x0 / np.sqrt(np.sum(np.square(x0)))  # normalize

print(f'initial RC guess: {x0}')

ret = optimize_rc(
    x0,
    metad_traj,
    sgoop_params,
    weights=weights,
    max_cal_traj=max_cal_traj,
    niter=100,
    annealing_temp=0.01,
    step_size=1.0
)
print(f'Best RC coeffient array: {", ".join([str(coeff) for coeff in ret.x])}', "\n")

plot_pmf(
    ret.x,
    metad_traj,
    sgoop_params,
    weights=weights,
    trial_rc=x0,
    normalize_grid=True,
    save_file='opt_rc_pmf.png'
)
