import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate


def gaussian_density_estimation(samples, weights, grid, h=0.1):
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
    h : float
        Bandwidth parameter for kernel density estimation. Associated with
        sigma in the case of a Gaussian kernel.
    Returns
    ----------
    np.ndarray
        The probability density values at the supplied grid points.
    """
    # KDE for fine-grained optimization
    kde = KDEUnivariate(samples)
    kde.fit(weights=weights, bw=h, fft=False)

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
        samples, weights=weights, bins=bins, density=True, range=hist_range
    )
    return hist, bin_edges


def find_closest_points(sequence, points):
    points_idx = np.zeros_like(sequence)
    for idx, val in enumerate(sequence):
        points_idx[idx] = np.abs(points - val).argmin()
    return points_idx


def avg_neighbor_transitions(sequence, num_neighbors):
    transitions = np.abs(sequence[1:] - sequence[:-1]) <= num_neighbors
    return np.sum(transitions) / (len(sequence) - 1)


def probability_matrix(p, d):
    prob_matrix = np.ones([len(p), len(p)]) * p
    multiplied = np.sqrt(prob_matrix * prob_matrix.T)
    denominator = 0
    divided = np.sqrt(prob_matrix / prob_matrix.T)
    matrix = np.zeros_like(divided)
    for idx in range(-d, d + 1):
        if idx != 0:
            # sum over multiplied offset axis for denominator
            denominator += np.trace(multiplied, offset=idx)
            # assign divided offset axis to matrix
            diag = np.diagonal(divided, offset=idx)
            matrix += np.diagflat(diag, k=idx)

    return denominator, matrix


def sorted_eigenvalues(matrix):
    # Returns eigenvalues, eigenvectors, and negative exponents of eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    eigenvalues.sort()  # Order eigenvalues
    return eigenvalues


def spectral_gap(eigen_values, wells):
    eigen_exp = np.exp(-eigen_values)
    gaps = eigen_exp[:-1] - eigen_exp[1:]

    if np.shape(gaps)[0] >= wells:
        return gaps[wells - 1]
    else:
        return 0
