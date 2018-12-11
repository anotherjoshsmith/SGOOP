import numpy as np
from sgoop.sgoop import md_prob
from sgoop.sgoop import bin_max_cal
from sgoop.sgoop import get_eigenvalues
from sgoop.sgoop import rc_eval
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
    binned = [0, 1, 2]
    prob = [
        0.4432692004460532,
        1.3298076013381424,
        2.216346002230199
    ]
    d = 1

    # test with binned traj, no diffusivity
    actual = get_eigenvalues(binned, prob, d, diffusivity=None)
    expected = [
        0,
        0.2647557475646271,
        0.6156877031058157,
    ]
    assert np.allclose(actual, expected)

    # assume static diffusivity, no maxcal trajectory
    actual = get_eigenvalues(None, prob, d, diffusivity=10)
    expected = [
        0,
        2.6475574756462716,
        6.156877031058157,
    ]
    assert np.allclose(actual, expected)

    # return None when binned traj and diffusivity are both None
    assert (get_eigenvalues(None, prob, d, diffusivity=None) is None)


def test_rc_eval():
    rc = np.array([0.3, 0.5, 0.8])
    samples = np.array(
        [[0, 1, 2],
         [3, 4, 0],
         [1, 2, 3]]
    )
    weights = [0.1, 0.3, 0.5]
    sgoop_dict = {
        'rc_bins': 3,
        'wells': 2,
        'd': 1,
        'diffusivity': 10,
    }

    # test with sgoop_dict and weights
    actual = rc_eval(rc, samples, sgoop_dict, weights=weights)
    expected = 0.0035751103510773745
    assert np.isclose(actual, expected)
    

def test_optimize_rc():
    assert True
