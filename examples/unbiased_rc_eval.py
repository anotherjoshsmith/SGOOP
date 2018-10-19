import numpy as np
import matplotlib.pyplot as plt

from sgoop.containers import load
from sgoop.sgoop import rc_eval


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
    thetas = np.linspace(0, 1, num=10) * np.pi
    # initialize dict of lists to store RCs and corresponding spectral gaps
    storage_lists = {
        'rc_list': None,
        'prob_list': None,
        'eigen_values_list': None,
        'sg_list': []
    }

    for idx, theta in enumerate(thetas):
        # assign reaction coordinate based on given theta
        single_sgoop.rc = np.array([np.sin(theta), np.cos(theta)])
        print(f'RC{idx:}: {theta:.3} rad')
        if storage_lists.get('rc_list') is not None:
            storage_lists['rc_list'].append(single_sgoop.rc)
        # evaluate given reaction coordinate
        rc_eval(single_sgoop, **storage_lists)
        print(f'spectral gap: {storage_lists["sg_list"][-1]:.4}')
        print()

    # plot the 2D free energy surface and best RC
    best_plot(single_sgoop, thetas, **storage_lists)
    plt.show()


def best_plot(single_scoop, thetas, **storage_dict):
    if not storage_dict.get('sg_list'):
        print('Ooops! Looks you forgot to store your Spectral Gaps. '
              'Please try again, with storage dictionary.')
        return

    sg_list = storage_dict['sg_list']

    data_array = single_scoop.max_cal_traj.values
    # Displays the best RC for 2D data
    best_rc = thetas[np.argmax(sg_list)]
    plt.figure()

    x_data = data_array[:, 0]
    y_data = data_array[:, 1]

    hist, x_edges, y_edges = np.histogram2d(x_data, y_data, 20)
    hist = hist.T
    prob = hist / np.sum(hist)
    energy = -np.log(prob)
    energy -= np.min(energy)

    # clip and center edges
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    x_edges = x_edges[:-1] + dx
    y_edges = y_edges[:-1] + dy

    # plot energy
    plt.contourf(x_edges, y_edges, energy)
    cbar = plt.colorbar()
    # plt.clim(0, clim)
    plt.set_cmap('viridis')
    cbar.ax.set_ylabel('G [kT]')
    # plot RC
    origin = [
        (x_data.max() + x_data.min()) / 2,
        (y_data.max() + y_data.min()) / 2,
    ]
    plt.title('Best RC = {0:.2f} rad'.format(best_rc))
    rcx = np.cos(best_rc)
    rcy = np.sin(best_rc)
    plt.quiver(*origin, rcx, rcy, scale=.1, color='grey');
    plt.quiver(*origin, -rcx, -rcy, scale=.1, color='grey');


if __name__ == '__main__':
    main()
