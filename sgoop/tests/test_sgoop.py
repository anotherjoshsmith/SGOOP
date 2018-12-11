import numpy as np
from sgoop.sgoop import md_prob
from sgoop.sgoop import bin_max_cal
# from sgoop.sgoop import get_eigenvalues
# from sgoop.sgoop import rc_eval
# from sgoop.sgoop import optimize_rc


def test_md_prob():
    rc = np.array([0.3, 0.5, 0.8])
    samples = np.array(
        [[0, 1, 2],
         [3, 4, 0],
         [1, 2, 3]]
    )

    # default weights = None, rc_bins = 20, kde_bw = None
    actual_prob, actual_grid = md_prob(rc, samples)
    expected_prob = [4.166666666666, 0.0, 0.0, 0.0, 0.0]
    expected_grid = [2.18, 2.26, 2.34, 2.42, 2.5]
    assert np.allclose(actual_prob[:5], expected_prob)
    assert np.allclose(actual_grid[:5], expected_grid)

    # add frame weights for the three observations, small rc_bins, add kde_bw
    weights = [0.1, 0.3, 0.5]
    actual_prob, actual_grid = md_prob(
        rc, samples, weights=weights, rc_bins=3, kde_bw=0.1
    )
    expected_prob = [
        0.4432692004460532,
        1.3298076013381424,
        2.216346002230199
    ]
    expected_grid = [2.1, 2.9, 3.7]
    assert np.allclose(actual_prob, expected_prob)
    assert np.allclose(actual_grid, expected_grid)


def test_bin_max_cal():
    rc = np.array([0.3, 0.5, 0.8])
    samples = np.array(
        [[0, 1, 2],
         [3, 4, 0],
         [1, 2, 3]]
    )
    grid = [2.1, 2.9, 3.7]

    # performing binning with rc and mdtraj
    actual = bin_max_cal(rc, samples, grid)
    expected = [0, 1, 2]
    assert np.allclose(actual, expected)

    # performing binning with rc and mdtraj
    assert (bin_max_cal(rc, None, grid) is None)
    assert (bin_max_cal(None, samples, grid) is None)


def test_get_eigenvalues():
    assert True


def test_rc_eval():
    assert True


def test_optimize_rc():
    assert True
