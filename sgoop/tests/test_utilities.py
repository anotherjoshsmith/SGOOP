import numpy as np
from sgoop.utilities import read_plumed_file
from sgoop.utilities import reweight_ct
from sgoop.utilities import calculate_sigma
# from sgoop.utilities import angle_to_rc


def test_read_plumed_file(tmp_path):
    # feed stringio that looks like COLVAR
    mock_colvar = "#! FIELDS time dist I384_dist\n 0.000 1.000 0.500"
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "COLVAR"
    p.write_text(mock_colvar)

    # test colvar columns, values, and index
    colvar = read_plumed_file(p)
    assert colvar.columns[1] == 'I384_dist'
    assert colvar['dist'].values == 1.000
    assert colvar.index[0] == 0.000


def test_reweight_ct():
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    actual = reweight_ct(weights, 2.5)
    expected = [
        1.0408107741923882,
        1.0832870676749586,
        1.1274968515793757,
        1.1735108709918103,
    ]
    assert np.allclose(actual, expected)


def test_calculate_sigma():
    samples = np.array(
        [[0, 1, 2],
         [3, 4, 0],
         [1, 2, 3]]
    )
    # test calculate sigma for all columns
    actual = calculate_sigma(samples)
    expected = [
        np.std([0, 3, 1]) / 2,
        np.std([0, 3, 1]) / 2,
        np.std([0, 3, 1]) / 2,
    ]
    assert np.allclose(actual, expected)

    # test calculate sigma for rc projection
    rc = np.array([0.3, 0.5, 0.8])
    actual = calculate_sigma(samples, rc)
    expected = 0.326598637109044
    assert np.allclose(actual, expected)


def test_angle_to_rc():
    assert True
