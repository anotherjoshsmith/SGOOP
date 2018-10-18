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


def main():
    data_dir = op.join(op.dirname(__file__), 'data')

    """User Defined Variables"""
    in_file = op.join(data_dir, 'trimmed.COLVAR')  # Input file
    rc_bin = 20  # Bins over RC
    wells = 2  # Expected number of wells with barriers > kT
    d = 1  # Distance between indexes for transition
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
        rc = [np.sin(theta), np.cos(theta)]
        spectral_gap[idx] = rc_eval(rc, data_array, rc_bin, wells,
                                    d, RC, P, SEE, SEV, SG)

    best_plot(data_array, RC, SG)
    plt.show()


def rei(SG, RC, P, SEE, SEV, SEVE):
    # Reinitializes arrays for new runs
    SG = []
    RC = []
    P = []
    SEE = []
    SEV = []
    SEVE = []


def normalize_rc(rc):
    # Normalizes input RC
    squares = 0
    for i in rc:
        squares += i ** 2
    denom = np.sqrt(squares)
    return np.array(rc) / denom


def generate_rc(i):
    # Generates a unit vector with angle pi*i
    x = np.cos(np.pi * i)
    y = np.sin(np.pi * i)
    return (x, y)


def md_prob(rc, data_array, rc_bin):
    # Calculates probability along a given RC
    proj = []

    for v in data_array:
        proj.append(np.dot(v, rc))
    rc_min = np.min(proj)
    rc_max = np.max(proj)
    binned = (proj - rc_min) / (rc_max - rc_min) * (rc_bin - 1)
    binned = np.array(binned).astype(int)

    prob = np.zeros(rc_bin)

    for point in binned:
        prob[point] += 1

    return prob / prob.sum(), binned  # Normalize


def set_bins(data_array, rc, bins, rc_min, rc_max):
    # Sets bins from an external source
    rc_bin = bins
    proj = np.dot(data_array, rc)
    binned = (proj - rc_min) / (rc_max - rc_min) * (rc_bin - 1)
    binned = np.array(binned).astype(int)  # mutating a global object...
    # return binned  # <--- better


def clean_whitespace(p, binned):
    # Removes values of imported data that do not match MaxCal data
    bmin = np.min(binned)
    bmax = np.max(binned)
    rc_bin = bmax - bmin + 1  # mutating a global object...
    binned -= bmin
    return p[bmin:bmax + 1]  # return p[bmin:bmax + 1], binned <--- better


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


def biased_prob(rc, old_rc, data_array, binned, rc_bin):
    # Calculates probabilities while "forgetting" original RC
    bias_prob = md_prob(old_rc)
    bias_bin = binned

    proj = []
    for v in data_array:
        proj.append(np.dot(v, rc))
    rc_min = np.min(proj)
    rc_max = np.max(proj)
    binned = (proj - rc_min) / (rc_max - rc_min) * (rc_bin - 1)
    binned = np.array(binned).astype(int)

    prob = np.zeros(rc_bin)

    for i in range(np.shape(binned)[0]):
        prob[binned[i]] += 1 / bias_prob[bias_bin[i]]  # Dividing by RAVE-like weights

    return prob / prob.sum()  # Normalize


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

    """Save RC for Calculations"""
    rc = normalize_rc(rc)
    RC.append(rc)

    """Probabilities and Index on RC"""
    prob, binned = md_prob(rc, data_array, rc_bin)
    P.append(prob)

    """Main SGOOP Method"""
    sg = sgoop(prob, binned, d, wells, rc_bin, SEV, SEE, SG)

    return sg


def biased_eval(rc, bias_rc, RC, P):
    # Biased SGOOP on a given RC with bias along a second RC
    # Input type: array of weights, probability from original RC

    """Save RC for Calculations"""
    rc = normalize_rc(rc)
    RC.append(rc)

    """Probabilities and Index on RC"""
    prob = biased_prob(rc, bias_rc)
    P.append(prob)

    """Main SGOOP Method"""
    sg = sgoop(rc, prob)

    return sg


if __name__ == '__main__':
    main()
