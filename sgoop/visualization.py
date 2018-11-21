import numpy as np
import matplotlib.pyplot as plt

from sgoop.sgoop import md_prob, rc_eval


def plot_spectral_gap(opt_rc, prob_traj, sgoop_dict, max_cal_traj=None, trial_rc=None):
    if max_cal_traj is None:
        max_cal_traj = prob_traj

    sg, eigenvalues = rc_eval(
        opt_rc, max_cal_traj, prob_traj, sgoop_dict, return_eigenvalues=True
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # plot
    plt.scatter(
        np.arange(eigenvalues), np.exp(-eigenvalues), label=f"spectral gap = {sg}"
    )

    if trial_rc is not None:
        sg, eigenvalues = rc_eval(
            opt_rc, max_cal_traj, prob_traj, sgoop_dict, return_eigenvalues=True
        )

        # plot
        plt.scatter(
            np.arange(eigenvalues), np.exp(-eigenvalues), label=f"spectral gap = {sg}"
        )

    return ax


def plot_pmf(opt_rc, prob_traj, sgoop_dict, trial_rc=None, normalize_grid=False):
    prob_args = (
        sgoop_dict["cv_cols"],
        sgoop_dict["v_minus_c_col"],
        sgoop_dict["rc_bins"],
        sgoop_dict["kde"],
    )

    prob, grid = md_prob(opt_rc, prob_traj, *prob_args)

    if normalize_grid:
        grid = (grid - grid.min()) / (grid.max() - grid.min())

    # initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # plot pmf from probability
    plt.plot(grid, -np.ma.log(prob), label="optimized RC")

    if trial_rc is not None:
        prob, grid = md_prob(trial_rc, prob_traj, *prob_args)

        if normalize_grid:
            grid = (grid - grid.min()) / (grid.max() - grid.min())

        # plot pmf from probability
        plt.plot(grid, -np.ma.log(prob), label="optimized RC")

    return ax
