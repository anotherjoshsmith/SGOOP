import numpy as np

import sys
sys.path.append('../../')

from sgoop.containers import load
from sgoop.sgoop import optimize_rc


# Specify the filenames for your biased and unbiased runs
metad_file = '../../sgoop/data/F399_COLVAR_8ns'  # biased colvar file
max_cal_file = '../../sgoop/data/max_cal.COLVAR'  # unbiased colvar file

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
    # create lists for storage, if ya want
    'rc_list': None,
    'prob_list': None,
    'ev_list': None,
    'sg_list': []
}

# load colvar files to
single_sgoop = load(max_cal_file, metad_file, **sgoop_params)
single_sgoop.max_cal_traj = single_sgoop.max_cal_traj.iloc[:5000, :]
single_sgoop.metad_traj = single_sgoop.metad_traj.iloc[::10, :]

# x0 = [1.21210308, -3.5540285, 0.57699017, 0.78250283,
#       0.67647444, 0.82403437, -0.45415575, -1.18827549,
#       9.0982063, -5.46860743, 3.47629007, 0.60513158,
#       0.85698247, 0.37547508, -0.79396719, -1.70581114]

# assign weight to biased CV from trial run
# np.random.seed(24)
x0 = np.array([0 for _ in range(16)])
x0[3] = 1
# x0 = x0 / np.sqrt(np.sum(np.square(x0)))  # normalize

print(f'initial RC guess: {x0}')

ret = optimize_rc(x0, single_sgoop, niter=100, annealing_temp=0.01, 
                  step_size=1.0)

print(f'Best RC coeffient array: {ret.x}')
