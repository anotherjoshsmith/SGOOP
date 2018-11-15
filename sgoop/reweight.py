"""This reweighting code is based on the algorithm proposed by Tiwary
and Parrinello, JPCB 2014, 119 (3), 736-742. This is a modified version
of te reweighting code based on earlier version (v1.0 - 23/04/2015) 
available in GitHub which was originally written by L. Sutto and 
F.L. Gervasio, UCL.

Co-Author: Debabrata Pramanik       pramanik@umd.edu
Co-Author: Zachary Smith            zsmith7@terpmail.umd.edu """

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate


def reweight(rc, metad_traj, cv_columns, v_minus_c_col, rc_bins=20, kt=2.5):
    """
    Reweighting biased MD trajectory to unbiased probabilty along a given reaction coordinate. Using rbias column from COLVAR to perform reweighting per Tiwary and Parinello

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
    kde.fit(weights=norm_weights, bw=0.05, fft=False)

    # evaluate pdf on a grid to for use in SGOOP
    grid = np.linspace(colvar_rc.min(), colvar.max(), num=rc_bins)
    pdf = kde.evaluate(grid)
    pdf = pdf / pdf.sum()

    return pdf, grid
