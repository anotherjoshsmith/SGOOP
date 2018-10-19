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
import scipy.optimize as opt


######################################################################
############ Get probabilities along RC with KDE #####################
######################################################################

def md_prob(rc, max_cal_traj, rc_bins, bandwidth=0.02, **storage_dict):
    # Calculates probability along a given RC
    data_array = max_cal_traj.values
    proj = np.sum(data_array * rc, axis=1)

    # get probability w/ statstmodels KDE
    kde = KDEUnivariate(proj)
    kde.fit(bw=bandwidth)

    grid = np.linspace(proj.min(), proj.max(), num=rc_bins)
    prob = kde.evaluate(grid)
    prob = prob / prob.sum()

    if storage_dict['prob_list'] is not None:
        storage_dict['prob_list'].append(prob)

    return prob, grid  # Normalize


def reweight(rc, metad_traj, cv_columns, v_minus_c_col,
             rc_bins=20, kt=2.5):
    """
    Reweighting biased MD trajectory to unbiased probabilty along
    a given reaction coordinate. Using rbias column from COLVAR to
    perform reweighting per Tiwary and Parinello

    """
    # read in parameters from sgoop object
    colvar = metad_traj[cv_columns].values
    v_minus_c = metad_traj[v_minus_c_col].values

    # calculate rc observable for each frame
    colvar_rc = np.sum(colvar * rc, axis=1)

    # calculate frame weights, per Tiwary and Parinello, JCPB 2015 (c(t) method)
    weights = np.exp(v_minus_c / kt)
    norm_weights = weights / weights.sum()

    # fit weighted KDE with statsmodels method
    kde = KDEUnivariate(colvar_rc)
    kde.fit(weights=norm_weights,
            bw=0.05,
            fft=False)

    # evaluate pdf on a grid to for use in SGOOP
    grid = np.linspace(colvar_rc.min(), colvar.max(), num=rc_bins)
    pdf = kde.evaluate(grid)
    pdf = pdf / pdf.sum()

    return pdf, grid


######################################################################
####### Get binned RC value along unbiased traj for MaxCal ###########
######################################################################

def bin_max_cal(rc, max_cal_traj, grid):
    # project unbiased observables onto
    proj = np.sum(max_cal_traj * rc, axis=1)

    binned = np.zeros_like(proj)
    for idx, val in enumerate(proj):
        binned[idx] = np.abs(grid - val).argmin()

    return binned


######################################################################
####### Calc transistion matrix from binned RC values from   #########
####### unbiased and probability from biased trajectory.     #########
######################################################################

def mu_factor(binned_rc_traj, p, d, rc_bins):
    # Calculates the prefactor on SGOOP for a given RC
    # Returns the mu factor associated with the RC
    # NOTE: mu factor depends on the choice of RC!
    # <N>, number of neighbouring transitions on each RC
    J = 0
    N_mean = 0
    for I in binned_rc_traj:
        N_mean += (np.abs(I - J) <= d) * 1
        J = np.copy(I)
    N_mean = N_mean / len(binned_rc_traj)

    D = 0
    for j in range(rc_bins):
        for i in range(rc_bins):
            if (np.abs(i - j) <= d) and (i != j):  # only count if we're neighbors?
                D += np.sqrt(p[j] * p[i])

    MU = N_mean / D
    return MU


def transmat(MU, p, d, rc_bins):
    # Generates transition matrix
    S = np.zeros([rc_bins, rc_bins])
    # Non diagonal terms
    # IFF the loop is necessary, we should build S and calculate D (aka MU) at the same time
    for j in range(rc_bins):
        for i in range(rc_bins):
            if (p[i] != 0) and (np.abs(i - j) <= d and (i != j)):
                S[i, j] = MU * np.sqrt(p[j] / p[i])

    """...we can now calculate the eigenvalues of the
    full transition matrix K, where Knm = −kmn for 
    m != n and Kmm = sum_m!=n(kmn)."""
    for i in range(rc_bins):
        # negate diagonal terms, which should be positive
        # after the next operation
        S[i, i] = -S.sum(1)[i]
    S = -np.transpose(S)  # negate and transpose

    return S


######################################################################
####### Calc eigenvalues and spectral gap from transition mat ########
######################################################################

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


def sgoop(p, binned, d, wells, rc_bins, **storage_dict):  # rc was never called
    # SGOOP for a given probability density on a given RC
    # Start here when using probability from an external source
    MU = mu_factor(binned, p, d, rc_bins)  # Calculated with MaxCal approach

    S = transmat(MU, p, d, rc_bins)  # Generating the transition matrix

    eigen_values, eigen_exp = eigeneval(S)  # Calculating eigenvalues and vectors for the transition matrix
    if storage_dict.get('eigen_value_list') is not None:
        storage_dict['eigen_value_list'].append(eigen_values)

    sg = spectral(wells, eigen_exp, eigen_values)  # Calculating the spectral gap
    if storage_dict.get('sg_list') is not None:
        storage_dict['sg_list'].append(sg)

    return sg


#####################################################################
####### Evaluate a series of RCs or optimize from starting RC #######
#####################################################################

def rc_eval(single_sgoop):
    # Unbiased SGOOP on a given RC
    rc = single_sgoop.rc
    max_cal_traj = single_sgoop.max_cal_traj
    rc_bins = single_sgoop.rc_bins
    wells = single_sgoop.wells
    d = single_sgoop.d
    storage_lists = single_sgoop.storage_dict

    """Save RC for Calculations"""  # why store in list? ahh, maybe for plotting?
    # normalize reaction coordinate vector
    rc = rc / np.sqrt(np.sum(np.square(rc)))

    """Probabilities and Index on RC"""
    # TODO: if biased, call biased prob (maybe write that within md_prob)
    prob, grid = md_prob(rc, max_cal_traj, rc_bins, **storage_lists)
    binned = bin_max_cal(rc, max_cal_traj, grid)

    """Main SGOOP Method"""
    sg = sgoop(prob, binned, d, wells, rc_bins, **storage_lists)

    return sg


def optimize_rc(rc_0, single_sgoop, niter=50, annealing_temp=0.01):
    """
    Calculate optimal RC given an initial estimate for the coefficients
    and a Sgoop object containing a COLVAR file with CVs tracked over
    the course of a short unbiased simulation a COLVAR file with
    c(t) and CVs from a biased MetaD simulation.

    :param rc_0:
    :param single_sgoop:
    :param niter:
    :param annealing_temp:
    :return:
    """
    minimizer_kwargs = {
        "options": {"maxiter": 10},
        "args": single_sgoop
    }

    return opt.basinhopping(__opt_func, rc_0,
                            niter=niter, T=annealing_temp, stepsize=1,
                            minimizer_kwargs=minimizer_kwargs,
                            callback=__print_fun)


def __opt_func(rc, single_sgoop):
    # unpack sgoop object for reweighting
    max_cal_traj = single_sgoop.max_cal_traj
    metad_traj = single_sgoop.metad_traj
    cv_cols = single_sgoop.cv_cols
    v_minus_c_col = single_sgoop.v_minus_c_col
    d = single_sgoop.d
    wells = single_sgoop.wells
    rc_bins = single_sgoop.rc_bins
    storage_dict = single_sgoop.storage_dict

    # calculate reweighted probability on RC grid
    prob, grid = reweight(rc, metad_traj, cv_cols,
                          v_minus_c_col, rc_bins)
    # get binned rc values from max cal traj
    binned_rc_traj = bin_max_cal(max_cal_traj, rc, grid)
    # calculate spectral gap for given rc and trajectories
    sg = sgoop(prob, binned_rc_traj, d, wells,
               rc_bins, **storage_dict)
    # return negative gap for minimization
    return -sg


def __print_fun(x, f, accepted):
    print(x, end=' ')
    if accepted == 1:
        print(f"with spectral gap {-f:.4} accepted.")
    else:
        print(f"with spectral gap {-f:.4} declined.")
