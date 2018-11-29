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
import scipy.optimize as opt
import sgoop.analysis as analysis


# #####################################################################
# ########### Get probabilities along RC with KDE #####################
# #####################################################################

def md_prob(rc, md_traj, weights=None, rc_bins=20, kde_bw=None):
    """
    Calculate probability density along a given reaction coordinate.

    Calculate the value of the probability density function (for KDE) or probability
    mass function (for histogram) at discrete grid points along a given RC. For a
    biased simulation, frame weights should be supplied.

    Parameters
    ----------
    rc : array-like
    md_traj : array-like
    weights : array-like, None
    rc_bins : int, 20
    kde_bw : float, None

    Returns
    -------
    pdf : np.ndarray
    grid : np.ndarray

    Examples
    --------


    """
    # ensure rc and md_traj are numpy arrays before computation
    rc = np.array(rc)
    md_traj = np.array(md_traj)
    # calculate rc observable for each frame
    colvar_rc = np.sum(md_traj * rc, axis=1)

    if kde_bw is not None:
        # evaluate pdf on a grid using KDE with Gaussian kernel
        grid = np.linspace(colvar_rc.min(), colvar_rc.max(), num=rc_bins)
        pdf = analysis.gaussian_density_estimation(colvar_rc, weights, grid, kde_bw)
        return pdf, grid

    # evaluate pdf using histograms
    pdf, bin_edges = analysis.histogram_density_estimation(
        colvar_rc, weights, rc_bins
    )
    # set grid points to center of bins
    bin_width = bin_edges[1] - bin_edges[0]
    grid = bin_edges[:-1] + bin_width

    return pdf, grid


# #####################################################################
# ###### Get binned RC value along unbiased traj for MaxCal ###########
# #####################################################################

def bin_max_cal(rc, md_traj, grid):
    """
    Calculate Reaction Coordinate bin index for each frame in max_cal_traj.

    Parameters
    ----------
    rc : np.ndarray
        Array of coefficients for one-dimensional reaction coordinate.
    md_traj : pd.DataFrame
        DataFrame storing COLVAR data from MaxCal trajectory.
    grid : np.ndarray
        Array of RC values at the center of each rc_bin.

    Returns
    ----------
    binned : np.ndarray

    """
    if rc is None or md_traj is None:
        return None

    # ensure rc and md_traj are ndarrays before computation
    rc = np.array(rc)
    md_traj = np.array(md_traj)
    # calculate rc observable for each frame
    colvar_rc = np.sum(md_traj * rc, axis=1)
    binned = analysis.find_closest_points(colvar_rc, grid)
    return binned


# #####################################################################
# ###### Calc transistion matrix from binned RC values from   #########
# ###### unbiased and probability from biased trajectory.     #########
# #####################################################################


def get_eigenvalues(binned_rc_traj, p, d, diffusivity=None):
    if diffusivity is None and binned_rc_traj is None:
        print('You must supply a MaxCal traj or diffusivity.')
        return

    n = diffusivity
    if binned_rc_traj is not None:
        # ensure binned traj is an ndarray before computation
        binned_rc_traj = np.array(binned_rc_traj)
        n = analysis.avg_neighbor_transitions(binned_rc_traj, d)

    with np.errstate(divide="ignore", invalid="ignore"):
        # ensure p is an ndarray before computation
        p = np.array(p)
        prob_matrix = analysis.probability_matrix(p, d)

    transition_matrix = n * prob_matrix
    eigenvalues = analysis.sorted_eigenvalues(transition_matrix)
    return eigenvalues


# #####################################################################
# ###### Calc eigenvalues and spectral gap from transition mat ########
# #####################################################################

def sgoop(p, binned, d, wells, diffusivity=None):
    # calculate eigenvalues and spectral gap
    eigen_values = get_eigenvalues(binned, p, d, diffusivity)
    sg = analysis.spectral_gap(eigen_values, wells)
    return sg


# ####################################################################
# ###### Evaluate a series of RCs or optimize from starting RC #######
# ####################################################################

def rc_eval(
        rc,
        probability_traj,
        sgoop_dict,
        weights=None,
        max_cal_traj=None,
        return_eigenvalues=False,
):
    # calculate prob for rc bins and binned rc value for MaxCal traj
    rc_bins = sgoop_dict.get('rc_bins')
    kde_bw = sgoop_dict.get('kde_bw')
    prob, grid = md_prob(rc, probability_traj, weights, rc_bins, kde_bw)
    # bin MaxCal trajectory. returns None if no trajectory is supplied.
    binned = bin_max_cal(rc, max_cal_traj, grid)
    # calculate spectral gap
    d = sgoop_dict.get('d')
    wells = sgoop_dict.get('wells')
    diffusivity = sgoop_dict.get('diffusivity')
    sg = sgoop(prob, binned, d, wells, diffusivity)

    if return_eigenvalues:
        eigenvalues = get_eigenvalues(binned, prob, d)
        return sg, eigenvalues

    return sg


def optimize_rc(
    rc_0,
    max_cal_traj,
    metad_traj,
    sgoop_dict,
    niter=50,
    annealing_temp=0.1,
    step_size=0.5,
):
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
    weights = None
    if sgoop_dict["v_minus_c_col"]:
        rbias = metad_traj[sgoop_dict["v_minus_c_col"]].values
        weights = analysis.reweight_ct(rbias)

    # pass trajectories and sgoop options through minimizer kwargs
    minimizer_kwargs = {
        "method": "BFGS",
        "options": {
            # "maxiter": 10
        },
        "args": (
            max_cal_traj,
            metad_traj,
            sgoop_dict["cv_cols"],
            weights,
            sgoop_dict["d"],
            sgoop_dict["wells"],
            sgoop_dict["rc_bins"],
            sgoop_dict["kde"],
            sgoop_dict["diffusivity"]
        ),
    }

    return opt.basinhopping(
        __opt_func,
        rc_0,
        niter=niter,
        T=annealing_temp,
        stepsize=step_size,
        minimizer_kwargs=minimizer_kwargs,
        disp=True,
        callback=__print_fun,
    )


def __opt_func(
    rc, max_cal_traj, metad_traj, cv_cols, weights, d, wells, rc_bins, kde, diffusivity,
):
    # normalize
    rc = rc / np.sqrt(np.sum(np.square(rc)))
    # calculate reweighted probability on RC grid
    prob, grid = md_prob(rc, metad_traj, cv_cols, weights, rc_bins, kde)
    # get binned rc values from max cal traj
    if max_cal_traj is not None:
        binned_rc_traj = bin_max_cal(rc, max_cal_traj, cv_cols, grid)
    else:
        binned_rc_traj = None
    # calculate spectral gap for given rc and trajectories
    sg = sgoop(prob, binned_rc_traj, d, wells, diffusivity)
    # return negative gap for minimization
    return -sg


def __print_fun(x, f, accepted):
    if accepted:
        rc = x / np.sqrt(np.sum(np.square(x)))
        print(f"RC with spectral gap {-f:} accepted.")
        print(", ".join([str(coeff) for coeff in rc]), "\n")
    else:
        print("")
