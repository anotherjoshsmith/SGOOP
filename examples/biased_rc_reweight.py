import matplotlib.pyplot as plt

from containers import load
from sgoop.reweight import reweight


def main():
    metad_file = '../sgoop/data/F399_dist.COLVAR'  # biased colvar file
    max_cal_file = '../sgoop/data/max_cal.COLVAR'  # unbiased colvar file

    sgoop_params = {
        'num_rc_bins': 100,
        'wells': 2,
        'd': 1,
    }

    single_sgoop = load(max_cal_file, metad_file, **sgoop_params)

    # define rbias and cv column names
    v_minus_c_col = 'metad.rbias'
    cv_columns = single_sgoop.metad_traj.columns[:16]

    single_sgoop.rc = [1 / len(cv_columns) for _ in cv_columns]

    pdf, grid, binned = reweight(single_sgoop, cv_columns, v_minus_c_col)
    
    # plot pdf on grid of 
    plt.plot(grid, pdf)
    plt.show()


if __name__ == '__main__':
    main()
