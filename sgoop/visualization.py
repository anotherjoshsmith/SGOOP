import numpy as np
import matplotlib.pyplot as plt


def eigenspectrum(sg, eigenvalues):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # plot
    plt.scatter(
        np.arange(eigenvalues),
        np.exp(-eigenvalues),
        label=f'spectral gap = {sg}'
    )

    return ax


def plot_pmf(prob, grid, normalize_grid=False):
    if normalize_grid:
        grid = (
            (grid - grid.min())
            / (grid.max() - grid.min())
        )

    # initialize plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # plot pmf from probability
    plt.plot(grid, -np.ma.log(prob),
             label='optimized RC')

    return ax
