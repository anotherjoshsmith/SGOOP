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
def md_prob(rc, md_traj, cv_columns, v_minus_c_col=None, rc_bins=20, kde=False, kt=2.5):
    """
    Calculate probability density along a given reaction coordinate.

    Reweighting biased MD trajectory to unbiased probabilty along
    a given reaction coordinate. Using rbias column from COLVAR to
    perform reweighting per Tiwary and Parinello

    Parameters
    ----------
    rc : int
    md_traj : pd.DataFrame
    cv_columns : List
    v_minus_c_col : str, None
    rc_bins : int, 20
    kde : bool, False
    kt : float, 2.5

    Returns
    -------
    pdf : np.ndarray
    grid : np.ndarray

    Examples
    --------


    """
    # read in parameters from sgoop object
    colvar = md_traj[cv_columns].values
    # calculate rc observable for each frame
    colvar_rc = np.sum(colvar * rc, axis=1)

    # calculate frame weights, per Tiwary and Parinello, JCPB 2015 (c(t) method)
    if v_minus_c_col:
        v_minus_c = md_traj[v_minus_c_col].values
        weights = np.exp(v_minus_c / kt)
        norm_weights = weights / weights.sum()
    else:
        norm_weights = None

    if kde:
        # evaluate pdf on a grid using KDE with Gaussian kernel
        grid = np.linspace(colvar_rc.min(), colvar_rc.max(), num=rc_bins)
        pdf = analysis.gaussian_density_estimation(colvar_rc, norm_weights, grid)
        return pdf, grid
    # evaluate pdf using histograms
    pdf, bin_edges = analysis.histogram_density_estimation(
        colvar_rc, norm_weights, rc_bins
    )
    # set grid points to center of bins
    bin_width = bin_edges[1] - bin_edges[0]
    grid = bin_edges[:-1] + bin_width

    return pdf, grid


# #####################################################################
# ###### Get binned RC value along unbiased traj for MaxCal ###########
# #####################################################################


def bin_max_cal(rc, md_traj, cv_columns, grid):
    """
    Calculate Reaction Coordinate bin index for each frame in max_cal_traj.

    Parameters
    ----------
    rc : np.ndarray
        Array of coefficients for one-dimensional reaction coordinate.
    max_cal_traj : pd.DataFrame
        DataFrame storing COLVAR data from MaxCal trajectory.
    grid : np.ndarray
        Array of RC values at the center of each rc_bin.

    Returns
    ----------
    binned : np.ndarray

    """
    # read in parameters from sgoop object
    max_cal_traj = md_traj[cv_columns].values
    # project unbiased observables onto
    proj = np.sum(max_cal_traj * rc, axis=1).values
    binned = analysis.find_closest_points(proj, grid)
    return binned


# #####################################################################
# ###### Calc transistion matrix from binned RC values from   #########
# ###### unbiased and probability from biased trajectory.     #########
# #####################################################################


def transition_matrix(binned_rc_traj, p, d, diffusivity=None):
    n = diffusivity
    if not n:
        n = analysis.avg_neighbor_transitions(binned_rc_traj, d)
    prob_matrix = analysis.probability_matrix(p, d)
    return n * prob_matrix  # negate and transpose


# #####################################################################
# ###### Calc eigenvalues and spectral gap from transition mat ########
# #####################################################################


def sgoop(p, binned, d, wells):
    # generate transition matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        trans_mat = transition_matrix(binned, p, d)
    # calculate eigenvalues and spectral gap
    eigen_values = analysis.sorted_eigenvalues(trans_mat)
    sg = analysis.spectral_gap(eigen_values, wells)
    return sg


# ####################################################################
# ###### Evaluate a series of RCs or optimize from starting RC #######
# ####################################################################
def rc_eval(max_cal_traj, sgoop_dict):
    # Unbiased SGOOP on a given RC
    rc = sgoop_dict["rc"]
    rc_bins = sgoop_dict["rc_bins"]
    wells = sgoop_dict["wells"]
    d = sgoop_dict["d"]
    cv_cols = sgoop_dict["cv_cols"]

    """Save RC for Calculations"""  # why store in list? ahh, maybe for plotting?
    # normalize reaction coordinate vector
    rc = rc / np.sqrt(np.sum(np.square(rc)))

    """Probabilities and Index on RC"""
    # TODO: if biased, call biased prob (maybe write that within md_prob)
    prob, grid = md_prob(rc, max_cal_traj, rc_bins)
    binned = bin_max_cal(rc, max_cal_traj, cv_cols, grid)

    """Main SGOOP Method"""
    sg = sgoop(prob, binned, d, wells)

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
            sgoop_dict["v_minus_c_col"],
            sgoop_dict["d"],
            sgoop_dict["wells"],
            sgoop_dict["rc_bins"],
            sgoop_dict["kde"],
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
    rc, max_cal_traj, metad_traj, cv_cols, v_minus_c_col, d, wells, rc_bins, kde
):
    # normalize
    rc = rc / np.sqrt(np.sum(np.square(rc)))
    # calculate reweighted probability on RC grid
    prob, grid = md_prob(rc, metad_traj, cv_cols, v_minus_c_col, rc_bins, kde)
    # get binned rc values from max cal traj
    binned_rc_traj = bin_max_cal(rc, max_cal_traj, cv_cols, grid)
    # calculate spectral gap for given rc and trajectories
    sg = sgoop(prob, binned_rc_traj, d, wells)
    # return negative gap for minimization
    return -sg


def __print_fun(x, f, accepted):
    if accepted:
        print(f"RC with spectral gap {-f:} accepted.")
        print(", ".join([str(coeff) for coeff in x]), "\n")
    else:
        print("")
