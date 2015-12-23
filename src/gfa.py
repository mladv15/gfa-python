"""
GFA (Group Factor Analysis)
This is a Python implementation of the file ./R/CCAGFA.R in the R package CCAGFA
"""

from __future__ import division, print_function


def gfa(Y, K,
        R="full", lmbda=0.1, rotate=True,
        opt_method="L-BFGS", opt_iter=10e5, lbfgs_factr=10e10, bfgs_crit=10e-5,
        init_tau=1000,
        iter_crit=10e-6, iter_max=10e5,
        addednoise=1e-5,
        prior_alpha_0=1e-14, prior_alpha_0t=1e-14,
        prior_beta_0=1e-14, prior_beta_0t=1e-14,
        dropK=True, low_mem=False,
        verbose=2):
    """
    They use getDefaultOpts() in the R package,
    but I guess specifying default argument values like this is more standard Python.
    """
    pass
