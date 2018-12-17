
import numpy as np
import matplotlib
from sgoop.visualization import plot_spectral_gap
from sgoop.visualization import plot_pmf

matplotlib.use('agg')


def test_plot_spectral_gap():
    rc = np.array([0.3, 0.5, 0.8])
    samples = np.array(
        [[0, 1, 2],
         [3, 4, 0],
         [1, 2, 3]]
    )
    sgoop_dict = {
        'rc_bins': 3,
        'wells': 2,
        'd': 1,
        'diffusivity': 10,
    }

    # test that plot_spectral_gap returns something
    ax = plot_spectral_gap(rc, samples, sgoop_dict)
    assert ax is not None


def test_plot_pmf():
    rc = np.array([0.3, 0.5, 0.8])
    samples = np.array(
        [[0, 1, 2],
         [3, 4, 0],
         [1, 2, 3]]
    )
    sgoop_dict = {
        'rc_bins': 3,
        'wells': 2,
        'd': 1,
        'diffusivity': 10,
    }

    # test that plot_pmf returns something
    ax = plot_pmf(rc, samples, sgoop_dict)
    assert ax is not None
