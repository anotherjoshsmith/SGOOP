import numpy as np
import matplotlib.pyplot as plt

from sgoop.containers import load
from sgoop.sgoop import rc_eval, best_plot


def main():
    """User Defined Variables"""
    sgoop_params = {
        'rc_bin': 20,
        'wells': 2,
        'd': 1,
    }

    # load colvar and SGOOP params into Sgoop object
    single_sgoop = load('../sgoop/data/trimmed.COLVAR', **sgoop_params)

    # generate array of theta values to populate trial 2D rxn coordinates
    thetas = np.linspace(0, 180, num=10)
    # initialize dict of lists to store RCs and corresponding spectral gaps
    storage_lists = {
        'rc_list': [],
        'prob_list': None,
        'eigen_values_list': None,
        'sg_list': []
    }

    for idx, theta in enumerate(thetas):
        # assign reaction coordinate based on given theta
        single_sgoop.rc = np.array([np.sin(theta), np.cos(theta)])
        print(f'RC{idx:}: {theta} deg')
        if storage_lists.get('rc_list') is not None:
            storage_lists['rc_list'].append(single_sgoop.rc)
        # evaluate given reaction coordinate
        rc_eval(single_sgoop, **storage_lists)
        print(f'spectral gap: {storage_lists["sg_list"][-1]}')
        print()

    # plot the 2D free energy surface and best RC
    best_plot(single_sgoop.colvar, **storage_lists)
    plt.show()


if __name__ == '__main__':
    main()
