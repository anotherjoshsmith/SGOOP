import numpy as np

import sys
sys.path.append('../../')

from sgoop.utilities import read_plumed_file, reweight_ct
from sgoop.sgoop import optimize_rc


# Specify the filenames for your biased and unbiased runs
max_cal_file = '../sgoop/data/max_cal.COLVAR'  # unbiased colvar file
metad_file = '../sgoop/data/F399_dist.COLVAR'  # biased colvar file

# specify collective variables and bias column (if applicable)
cv_cols = [
    'dist', 'I384_dist', 'N387_dist', 'F399_dist', 'L403_dist', 'V429_dist',
    'A445_dist', 'L449_dist', 'R406_dist.min', 'Y407_dist.min', 'L426_dist.min',
    'R481_dist.min', 'S485_dist.min', 'I384_R406', 'I384_S485', 'R406_S485'
]
rbias_col = 'metad.rbias'
# load MaxCal dataframe from PLUMED colvar file
max_cal_traj = read_plumed_file(max_cal_file, cv_columns=cv_cols)
# load metad CV dataframe and rbias series (for c(t) reweighting
metad_traj, rbias = read_plumed_file(
    metad_file, cv_columns=cv_cols, bias_column=rbias_col
)

# take first 5000 entries of MaxCal to speed up computation
max_cal_traj = max_cal_traj.iloc[:5000, :]
# likewise, take every tenth entry from MetaD
metad_traj = metad_traj.iloc[::10, :]
rbias = rbias.iloc[::10]

# calculate frame weights from rbias
weights = reweight_ct(rbias, kt=2.5)

# specify columns you want to require
sgoop_params = {
    # adjust sgoop params
    'rc_bins': 20,
    # 'kde_bw': 0.01,
    'd': 1,
    'wells': 2,
    'diffusivity': None,
}


# assign weight to biased CV from trial run
# np.random.seed(24)
x0 = np.array([0 for _ in range(16)])
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
