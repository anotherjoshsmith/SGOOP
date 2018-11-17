import numpy as np

import sys
sys.path.append('../../')

from sgoop.containers import read_plumed_file
from sgoop.sgoop import optimize_rc


# Specify the filenames for your biased and unbiased runs
max_cal_file = '../sgoop/data/max_cal.COLVAR'  # unbiased colvar file
metad_file = '../sgoop/data/F399_dist.COLVAR'  # biased colvar file
# load colvar files to
max_cal_traj = read_plumed_file(max_cal_file).iloc[:5000, :]
metad_traj = read_plumed_file(metad_file).iloc[::10, :]


# specify columns you want to require
sgoop_params = {
    # specify rbias and cv column names
    'v_minus_c_col': 'metad.rbias',
    'cv_cols': ['dist', 'I384_dist', 'N387_dist', 'F399_dist',
                'L403_dist', 'V429_dist', 'A445_dist', 'L449_dist',
                'R406_dist.min', 'Y407_dist.min', 'L426_dist.min',
                'R481_dist.min', 'S485_dist.min', 'I384_R406',
                'I384_S485', 'R406_S485'],
    # adjust sgoop params
    'rc_bins': 20,
    'wells': 2,
    'd': 1,
    'kde': False,
}


# assign weight to biased CV from trial run
# np.random.seed(24)
x0 = np.array([0 for _ in range(16)])
x0[3] = 1
# x0 = x0 / np.sqrt(np.sum(np.square(x0)))  # normalize

print(f'initial RC guess: {x0}')

ret = optimize_rc(x0, max_cal_traj, metad_traj, sgoop_params, niter=100,
                  annealing_temp=0.01, step_size=1.0)

print(f'Best RC coeffient array: {ret.x}')
