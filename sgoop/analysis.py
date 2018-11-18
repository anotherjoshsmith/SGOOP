import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate


def gaussian_density_estimation(samples, weights, grid):
    """
    Kernel density estimation with Gaussian kernel.


    Parameters
    ----------
    samples : np.ndarray
        Array of sample values.
    weights : np.ndarray
        Array of sample weights. If None, unweighted KDE will be performed.
    grid : np.ndarray
        Grid points at which the KDE function should be evaluated.

    Returns
    ----------
    np.ndarray
        The probability density values at the supplied grid points.
    """
    # KDE for fine-grained optimization
    kde = KDEUnivariate(samples)
    kde.fit(weights=weights, bw=0.1, fft=False)

    # evaluate pdf on a grid to for use in SGOOP
    # TODO: area under curve between points instead of pdf at point
    return kde.evaluate(grid)


def histogram_density_estimation(samples, weights, bins):
    """
    Kernel density estimation with Gaussian kernel.


    Parameters
    ----------
    samples : np.ndarray
        Array of sample values.
    weights : np.ndarray
        Array of sample weights. If None, unweighted KDE will be performed.
    bins : int
        Number of bins used for histogramming

    Returns
    ----------
    hist : np.ndarray
        The probability density values for each bin.
    bin_edges : np.ndarray
        The edges of each bin.
    """
    # histogram density for coarse optimization
    hist_range = (samples.min(), samples.max())
    hist, bin_edges = np.histogram(
        samples,
        weights=weights,
        bins=bins,
        density=True,
        range=hist_range,
    )
    return hist, bin_edges


def find_closest_points(sample, points):
    binned = np.zeros_like(sample)
    for idx, val in enumerate(sample):
        binned[idx] = np.abs(points - val).argmin()

    return binned