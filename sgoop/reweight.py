"""This reweighting code is based on the algorithm proposed by Tiwary
and Parrinello, JPCB 2014, 119 (3), 736-742. This is a modified version
of te reweighting code based on earlier version (v1.0 - 23/04/2015) 
available in GitHub which was originally written by L. Sutto and 
F.L. Gervasio, UCL.

Co-Author: Debabrata Pramanik       pramanik@umd.edu
Co-Author: Zachary Smith            zsmith7@terpmail.umd.edu """

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate


def reweight(single_sgoop, cv_columns, v_minus_c_col, kt=2.5):
    """
    Reweighting biased MD trajectory to unbiased probabilty along
    a given reaction coordinate. Using rbias column from COLVAR to
    perform reweighting per Tiwary and Parinello

    :param single_sgoop:
    :param cv_columns:
    :param v_minus_c_col:
    :param kt:
    :return:
    """

    # read in parameters from sgoop object
    colvar = single_sgoop.metad_traj[cv_columns].values
    v_minus_c_col = single_sgoop.metad_traj[v_minus_c_col].values
    rc = single_sgoop.rc
    num_rc_bins = single_sgoop.num_rc_bins

    # calculate rc observable for each frame
    colvar_rc = np.sum(colvar * rc, axis=1)

    # calculate frame weights, per Tiwary and Parinello, JCPB 2015 (c(t) method)
    weights = np.exp(v_minus_c_col / kt)
    norm_weights = weights / weights.sum()

    # fit weighted KDE with statsmodels method
    kde = KDEUnivariate(colvar_rc)
    kde.fit(weights=norm_weights,
            bw=0.05,
            fft=False)

    # evaluate pdf on a grid to for use in SGOOP
    grid = np.linspace(colvar_rc.min(), colvar.max(), num=num_rc_bins)
    pdf = kde.evaluate(grid)
    pdf = pdf / pdf.sum()

    # get max_cal transition bins
    binned = ((colvar_rc - colvar_rc.min())
              / (np.ptp(colvar_rc))  # normalize
              * (num_rc_bins - 1))  # multiply by number of bins
    binned = binned.astype(int)

    return pdf, grid, binned
