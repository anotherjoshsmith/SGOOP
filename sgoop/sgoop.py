""" Script that evaluates reaction coordinates using the SGOOP method. 
Probabilites are calculated using MD trajectories. Transition rates are
found using the maximum caliber approach.  
For unbiased simulations use rc_eval().
For biased simulations calculate unbiased probabilities and analyze then with sgoop().

The original method was published by Tiwary and Berne, PNAS 2016, 113, 2839.

Author: Zachary Smith                   zsmith7@terpmail.umd.edu
Original Algorithm: Pratyush Tiwary     ptiwary@umd.edu 
Contributor: Pablo Bravo Collado        ptbravo@uc.cl"""

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate


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

    return pdf / pdf.sum()


def md_prob(rc, max_cal_traj, rc_bin, bandwidth=0.02, **storage_dict):
    # Calculates probability along a given RC
    data_array = max_cal_traj.values
    proj = np.sum(data_array * rc, axis=1)

    binned = ((proj - proj.min()) / (np.ptp(proj))  # normalize
              * (rc_bin - 1))  # multiply by number of bins
    binned = binned.astype(int)

    # ###################################
    # ########## METHOD ONE #############
    # ###################################
    # get probability w/ my KDE (SLOWER)
    # m = proj.shape[0] // np.sqrt(proj.shape[0])  <--- This method is much slower, but also much more accurate.
    # m = rc_bin
    # grid = np.linspace(proj.min() - 3 * proj.std(),
    #                    proj.max() + 3 * proj.std(),
    #                    num=m)
    # prob = density_estimation(proj, grid, bandwidth=0.02)

    # ###################################
    # ########## METHOD TWO #############
    # ###################################
    # get probability w/ statstmodels KDE
    m = rc_bin
    grid = np.linspace(proj.min(), proj.max(), num=m)
    kde = KDEUnivariate(proj)
    kde.fit(bw=bandwidth)
    prob = kde.evaluate(grid)
    prob = prob / prob.sum()

    if storage_dict['prob_list'] is not None:
        storage_dict['prob_list'].append(prob)

    return prob, binned  # Normalize


def mu_factor(binned, p, d, rc_bin):
    # Calculates the prefactor on SGOOP for a given RC
    # Returns the mu factor associated with the RC
    # NOTE: mu factor depends on the choice of RC!
    # <N>, number of neighbouring transitions on each RC
    J = 0
    N_mean = 0
    for I in binned:
        N_mean += (np.abs(I - J) <= d) * 1
        J = np.copy(I)
    N_mean = N_mean / len(binned)


    # TODO: get rid of double-loop
    D = 0
    for j in range(rc_bin):
        for i in range(rc_bin):
            if (np.abs(i - j) <= d) and (i != j):  # only count if we're neighbors?
                D += np.sqrt(p[j] * p[i])

    MU = N_mean / D
    return MU


def transmat(MU, p, d, rc_bin):
    # Generates transition matrix
    S = np.zeros([rc_bin, rc_bin])
    # Non diagonal terms
    # IFF the loop is necessary, we should build S and calculate D (aka MU) at the same time
    for j in range(rc_bin):
        for i in range(rc_bin):
            if (p[i] != 0) and (np.abs(i - j) <= d and (i != j)):
                S[i, j] = MU * np.sqrt(p[j] / p[i])

    """...we can now calculate the eigenvalues of the
    full transition matrix K, where Knm = −kmn for 
    m != n and Kmm = sum_m!=n(kmn)."""
    for i in range(rc_bin):
        # negate diagonal terms, which should be positive
        # after the next operation
        S[i, i] = -S.sum(1)[i]
    S = -np.transpose(S)  # negate and transpose

    return S


def eigeneval(matrix):
    # Returns eigenvalues, eigenvectors, and negative exponents of eigenvalues
    eigenValues, eigenVectors = np.linalg.eig(matrix)
    idx = eigenValues.argsort()  # Sorting by eigenvalues
    eigenValues = eigenValues[idx]  # Order eigenvalues
    eigenExp = np.exp(-eigenValues)  # Calculate exponentials
    return eigenValues, eigenExp


def spectral(wells, eigen_exp, eigen_values):
    SEE_pos = eigen_exp[(eigen_values > -1e-10)]  # Removing negative eigenvalues
    SEE_pos = SEE_pos[SEE_pos > 0]  # Removing negative exponents
    gaps = SEE_pos[:-1] - SEE_pos[1:]
    if np.shape(gaps)[0] >= wells:
        return gaps[wells - 1]
    else:
        return 0


def sgoop(p, binned, d, wells, rc_bin, **storage_dict):  # rc was never called
    # SGOOP for a given probability density on a given RC
    # Start here when using probability from an external source
    MU = mu_factor(binned, p, d, rc_bin)  # Calculated with MaxCal approach

    S = transmat(MU, p, d, rc_bin)  # Generating the transition matrix

    eigen_values, eigen_exp = eigeneval(S)  # Calculating eigenvalues and vectors for the transition matrix
    if storage_dict.get('eigen_value_list') is not None:
        storage_dict['eigen_value_list'].append(eigen_values)

    sg = spectral(wells, eigen_exp, eigen_values)  # Calculating the spectral gap
    if storage_dict.get('sg_list') is not None:
        storage_dict['sg_list'].append(sg)

    return sg


def rc_eval(single_sgoop, **storage_lists):
    # Unbiased SGOOP on a given RC
    # Input type: array of weights

    rc = single_sgoop.rc
    max_cal_traj = single_sgoop.max_cal_traj
    rc_bin = single_sgoop.rc_bin
    wells = single_sgoop.wells
    d = single_sgoop.d

    """Save RC for Calculations"""  # why store in list? ahh, maybe for plotting?
    # normalize reaction coordinate vector
    rc = rc / np.sqrt(np.sum(np.square(rc)))

    """Probabilities and Index on RC"""
    # TODO: if biased, call biased prob (maybe write that within md_prob)
    prob, binned = md_prob(rc, max_cal_traj, rc_bin, **storage_lists)

    """Main SGOOP Method"""
    sg = sgoop(prob, binned, d, wells, rc_bin, **storage_lists)

    return sg
