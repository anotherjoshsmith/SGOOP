""" Script that evaluates reaction coordinates using the SGOOP method. 
Probabilites are calculated using MD trajectories. Transition rates are
found using the maximum caliber approach.  
For unbiased simulations use rc_eval().
For biased simulations calculate unbiased probabilities and analyze then with sgoop().

The original method was published by Tiwary and Berne, PNAS 2016, 113, 2839.

Author: Zachary Smith                   zsmith7@terpmail.umd.edu
Original Algorithm: Pratyush Tiwary     ptiwary@umd.edu 
Contributor: Pablo Bravo Collado        ptbravo@uc.cl"""

import os.path as op

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix


# TODO: SGOOP container class...
class Sgoop:
    def __init__(self, colvar, rc_bin=20, wells=2, d=1, SG=[], RC=[], P=[], SEE=[], SEV=[]):
        pass


def main():
    data_dir = op.join(op.dirname(__file__), 'data')

    """User Defined Variables"""
    in_file = op.join(data_dir, 'trimmed.COLVAR')  # Input file
    rc_bin = 20  # Bins over RC
    wells = 2  # Expected number of wells with barriers > kT
    d = 1  # Distance between indexes for transition <- TODO: ask what this is. will it ever be anything but 1?
    # prob_cutoff = 1e-5  # Minimum nonzero probability

    """Auxiliary Variables"""
    SG = []  # List of Spectral Gaps
    RC = []  # List of Reaction Coordinates
    P = []  # List of probabilites on RC
    SEE = []  # SGOOP Eigen exp
    SEV = []  # SGOOP Eigen values
    # SEVE = []  # SGOOP Eigen vectors

    """Load MD File"""
    data_array = np.loadtxt(in_file)[:, :]

    # ##### PRACTICE SCRIPT
    thetas = np.linspace(0, 180, num=100)
    spectral_gap = np.zeros_like(thetas)

    for idx, theta in enumerate(thetas):
        rc = np.array([np.sin(theta), np.cos(theta)])
        spectral_gap[idx] = rc_eval(rc, data_array, rc_bin, wells,
                                    d, RC, P, SEE, SEV, SG)

    best_plot(data_array, RC, SG)
    plt.show()


def density_estimation(x, grid, bandwidth=0.02):
    # need to adapt to the 2D case (esp. for visualization)
    d = 1
    if len(x.shape) > 1:
        d = x.shape[1]

    pdf = np.zeros_like(grid)
    for idx, pt in enumerate(grid):
        dists = x - pt
        prefactor = np.power(2 * np.pi * np.power(bandwidth, 2.), (- d / 2.))
        gaussians = prefactor * (np.exp(-np.power(dists, 2.)
                                 / (2 * np.power(bandwidth, 2.))))

        pdf[idx] = np.sum(gaussians)

    return pdf


def md_prob(rc, data_array, rc_bin):
    # Calculates probability along a given RC
    proj = np.sum(data_array * rc, axis=1)

    rc_min = proj.min()
    rc_max = proj.max()
    binned = (proj - rc_min) / (rc_max - rc_min) * (rc_bin - 1)
    binned = np.array(binned).astype(int)

    # get probability w/ KDE
    m = proj.shape[0] // np.sqrt(proj.shape[0])

    grid = np.linspace(proj.min() - 3 * proj.std(),
                       proj.max() + 3 * proj.std(),
                       num=m)

    prob = density_estimation(proj, grid, bandwidth=0.02)

    return prob / prob.sum(), binned  # Normalize


def eigeneval(matrix):
    # Returns eigenvalues, eigenvectors, and negative exponents of eigenvalues
    eigenValues, eigenVectors = np.linalg.eig(matrix)
    idx = eigenValues.argsort()  # Sorting by eigenvalues
    eigenValues = eigenValues[idx]  # Order eigenvalues
    eigenVectors = eigenVectors[:, idx]  # Order eigenvectors
    eigenExp = np.exp(-eigenValues)  # Calculate exponentials
    return eigenValues, eigenExp, eigenVectors


def mu_factor(binned, p, d, rc_bin):
    # Calculates the prefactor on SGOOP for a given RC
    # Returns the mu factor associated with the RC
    # NOTE: mu factor depends on the choice of RC!
    # <N>, number of neighbouring transitions on each RC
    J = 0
    N_mean = 0
    D = 0
    for I in binned:
        N_mean += (np.abs(I - J) <= d) * 1
        J = np.copy(I)
    N_mean = N_mean / len(binned)

    # Denominator
    for j in range(rc_bin):
        for i in range(rc_bin):
            if (np.abs(i - j) <= d) and (i != j):
                D += np.sqrt(p[j] * p[i])
    MU = N_mean / D
    return MU


def transmat(MU, p, d, rc_bin):
    # Generates transition matrix
    S = np.zeros([rc_bin, rc_bin])
    # Non diagonal terms
    for j in range(rc_bin):
        for i in range(rc_bin):
            if (p[i] != 0) and (np.abs(i - j) <= d and (i != j)):
                S[i, j] = MU * np.sqrt(p[j] / p[i])

    for i in range(rc_bin):
        S[i, i] = -S.sum(1)[i]  # Diagonal terms
    S = -np.transpose(S)  # Tranpose and fix

    return S


def spectral(wells, SEE, SEV):
    # Calculates spectral gap for appropriate number of wells
    SEE_pos = SEE[-1][SEV[-1] > -1e-10]  # Removing negative eigenvalues
    SEE_pos = SEE_pos[SEE_pos > 0]  # Removing negative exponents
    gaps = SEE_pos[:-1] - SEE_pos[1:]
    if np.shape(gaps)[0] >= wells:
        return gaps[wells - 1]
    else:
        return 0


def sgoop(p, binned, d, wells, rc_bin, SEV, SEE, SG):  # rc was never called
    # SGOOP for a given probability density on a given RC
    # Start here when using probability from an external source
    MU = mu_factor(binned, p, d, rc_bin)  # Calculated with MaxCal approach

    S = transmat(MU, p, d, rc_bin)  # Generating the transition matrix

    sev, see, seve = eigeneval(S)  # Calculating eigenvalues and vectors for the transition matrix
    SEV.append(sev)  # Recording values for later analysis
    SEE.append(see)
    # SEVE.append(seve)

    sg = spectral(wells, SEE, SEV)  # Calculating the spectral gap
    SG.append(sg)

    return sg


def best_plot(data_array, RC, SG):
    # Displays the best RC for 2D data
    best_rc = np.ceil(np.arccos(RC[np.argmax(SG)][0]) * 180 / np.pi)
    plt.figure()
    cmap = plt.cm.get_cmap("jet")
    hist = np.histogram2d(data_array[:, 0], data_array[:, 1], 20)
    hist = hist[0]
    prob = hist / np.sum(hist)
    potE = -np.ma.log(prob)
    potE -= np.min(potE)
    np.ma.set_fill_value(potE, np.max(potE))
    plt.contourf(np.transpose(np.ma.filled(potE)), cmap=cmap)

    plt.title('Best RC = {0:.2f} Degrees'.format(best_rc))
    origin = [10, 10]
    rcx = np.cos(np.pi * best_rc / 180)
    rcy = np.sin(np.pi * best_rc / 180)
    plt.quiver(*origin, rcx, rcy, scale=.1, color='grey');
    plt.quiver(*origin, -rcx, -rcy, scale=.1, color='grey');


def rc_eval(rc, data_array, rc_bin, wells, d, RC, P, SEE, SEV, SG):
    # Unbiased SGOOP on a given RC
    # Input type: array of weights

    """Save RC for Calculations"""  # why store in list? ahh, maybe for plotting?
    # normalize reaction coordinate vector
    rc = rc / np.sqrt(np.sum(np.square(rc)))
    RC.append(rc)

    """Probabilities and Index on RC"""
    # TODO: if biased, call biased prob (maybe write that within md_prob)
    prob, binned = md_prob(rc, data_array, rc_bin)
    P.append(prob)

    """Main SGOOP Method"""
    sg = sgoop(prob, binned, d, wells, rc_bin, SEV, SEE, SG)

    return sg


if __name__ == '__main__':
    main()
