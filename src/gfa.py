"""
GFA (Group Factor Analysis)
This is a Python implementation of the file ./R/CCAGFA.R in the R package CCAGFA
"""

from __future__ import division, print_function
import numpy as np


def gfa_experiments(Y, K, Nrep=10, verbose=2, **opts):
    """
    A wrapper for running the GFA model `Nrep` times
    and choosing the final model based on the best
    lower bound. This is the recommended way of applying
    the algorithm.
    See GFA() for description of the inupts.
    """
    # TODO: this is just a placeholder, will add real values after gfa() is finished
    opts["verbose"] = verbose
    lb = []  # lower bounds
    models = []  # the best one will be returned
    for rep in range(Nrep):
        model = gfa(Y, K, **opts)
        models.append(model)
        # lb.append(model.cost)  # not defined yet
        if verbose == 1:
            print("Run %d/%d: %d iterations with final cost %f") % (rep, Nrep, 1337, 1338)
    # uncomment below when done
    # k = np.argmax(lb)
    # return models[k]


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
    Parameters
    ----------
    Y : list
       List of M data ndarrays. Y[m] is an ndarray (matrix) with
       N rows (samples) and D_m columns (features). The
       samples need to be co-occurring.
       NOTE: All of these should be centered, so that the mean
             of each feature is zero
       NOTE: The algorithm is roughly invariant to the scale
             of the data, but extreme values should be avoided.
             Data with roughly unit variance or similar scale
             is recommended.
    K : int
        The number of components

    Returns
    -------
    The trained model, which is a dict that contains the following elements:
    TODO: (could make the model an object later)
        Z    : The mean of the latent variables; N times K matrix
        covZ : The covariance of the latent variables; K times K matrix
        ZZ   : The second moments ZZ^T; K times K matrix

        W    : List of the mean projections; D_i times K matrices
        covW : List of the covariances of the projections; D_i times D_i matrices
        WW   : List of the second moments WW^T; K times K matrices

        tau  : The mean precisions (inverse variance, so 1/tau gives the
            variances denoted by sigma in the paper); M-element vector

        alpha: The mean precisions of the projection weights, the
            variances of the ARD prior; M times K matrix

        U,V,u.mu,v.mu: The low-rank factorization of alpha.

        cost : Vector collecting the variational lower bounds for each
            iteration
        D    : Data dimensionalities; M-element vector
        datavar   : The total variance in the data sets, needed for
                 GFAtrim()
        addednoise: The level of extra noise as in opts$addednoise

    They use getDefaultOpts() in the R package,
    but I guess specifying default argument values like this is more standard Python,
    like scikit learn https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/gradient_boosting.py#L723.
    """
    # check that data is centered
    for m, Y_m in enumerate(Y):
        if not np.all(np.abs(np.mean(Y_m, axis=1)) < 1e-7):
            print("Warning: data from group %d does not have zero mean" % m)

    # check that there is more than one group of data
    if len(Y) < 2:
        print("Warning: the number of data sets must be larger than 1")

    # store dimensions
    M = len(Y)
    D = [Y_m.shape[1] for Y_m in Y]  # D = [D_1, ..., D_M]
    Ds = sum(D)                      # total nr of features
    N = Y[0].shape[0]                # total number of samples
