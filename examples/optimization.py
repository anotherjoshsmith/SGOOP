from sgoop.containers import load
from sgoop.sgoop import optimize_rc


# Specify the filenames for your biased and unbiased runs
metad_file = '../sgoop/data/F399_dist.COLVAR'  # biased colvar file
max_cal_file = '../sgoop/data/max_cal.COLVAR'  # unbiased colvar file

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
    'rc_bins': 100,
    'wells': 2,
    'd': 1,
    # create lists for storage, if ya want
    'rc_list': [],
    'prob_list': [],
    'ev_list': [],
    'sg_list': []
}

# load colvar files to
single_sgoop = load(max_cal_file, metad_file, **sgoop_params)

x0 = [1 / 16 for i in range(16)]
ret = optimize_rc(x0, single_sgoop, niter=5)

print(f'Best RC coeffient array: {ret.x}')
