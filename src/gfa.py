"""
GFA (Group Factor Analysis)
This is a Python implementation of the file ./R/CCAGFA.R in the R package CCAGFA
"""

from __future__ import division, print_function
import numpy as np
from scipy import special
import math


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
    D = [Y_m.shape[1] for Y_m in Y]  # Data dimensions for each group. D = [D_1, ..., D_M]
    D = np.array(D)
    Ds = sum(D)                      # total nr of features
    N = Y[0].shape[0]                # total number of samples
    datavar = []                     # total variance of the data for each group
    for Y_m in Y:
        # Y_m is NxD_m, so take variance along column (axis=0), total variance <- sum
        datavar.append(sum(np.var(Y_m, axis=0)))

    if isinstance(R, int) and R >= min(M, K):
        if verbose == 2:
            print("The rank corresponds to full rank solution.")
        R = "full"
    if R != "full":
        if verbose == 2:
            print("NOTE: optimization of the rotation is not supported for low rank model")
        rotate = False

    # Some constants for speeding up the computation
    const = N*Ds/2*np.log(2*np.pi)  # constant factors for the lower bound
    Yconst = [np.sum(np.vectorize(pow)(Y_m, 2)) for Y_m in Y]
    id1 = np.ones(K)
    alpha_0 = prior_alpha_0   # Easier access for hyperprior values
    beta_0 = prior_beta_0
    alpha_0t = prior_alpha_0t
    beta_0t = prior_beta_0t

    #
    # Initialize the model randomly; other initializations could
    # be done, but overdispersed random initialization is quite good.
    #

    # Latent variables Z
    Z = np.random.randn(N, K)   # The mean
    covZ = np.diag(np.ones(K))  # The covariance
    ZZ = covZ + covZ*N          # The second moments

    # ARD and noise parameters (What is ARD?)
    alpha = np.ones((M, K))     # The mean of the ARD precisions
    logalpha = np.ones((M, K))  # The mean of <\log alpha>
    if R=="full":
        b_ard = np.ones((M, K))     # The parameters of the Gamma distribution
        a_ard = alpha_0 + D/2       #       for ARD precisions
        # psi is digamma, derivative of the logarithm of the gamma function
        digammaa_ard = special.psi(a_ard)
    tau = np.repeat(init_tau, M)    # The mean noise precisions
    a_tau = alpha_0t + N*D/2        # The parameters of the Gamma distribution
    b_tau = np.zeros(M)             #   for the noise precisions
    digammaa_tau = special.psi(a_tau)  # Constants needed for computing the lower bound
    lgammaa_tau = -np.sum(np.vectorize(math.lgamma)(a_tau))
    lb_pt_const = -M*np.vectorize(math.lgamma)(alpha_0t) + M*alpha_0t*np.log(beta_0t)

    # Alpha needs to be initialized to match the data scale
    for m in range(M):
        alpha[m, :] = K*D[m]/(datavar[m]-1/tau[m])

    # The projections
    # No need to initialize projections randomly ,since their updating
    # step is the first one; just define the variables here
    low_mem = True
    W = [None]*M  # the means
    if not low_mem:
        covW = [None]*M  # the covariances
    else: 
        covW = np.diag(np.ones(K))

    WW = [None]*M  # the second moments
    for m in range(M):
        # I think the more standard way would be to let W[m] be KxD_m
        # but they apparently set it to (D_m x K)
        W[m] = np.zeros((D[m], K))  # So each W[m] is actually W[m].T
        if not low_mem:
            covW[m] = np.diag(np.ones(K))
            # matrix crossproduct of W is W.T %*% W
            WW[m] = np.dot(W[m].T, W[m]) + covW[m]*D[m]
        else:
            WW[m] = np.dot(W[m].T, W[m]) + covW*D[m]

    # TODO: These are for alpha, U, and V
    # Rotation parameters
    
    # Use R-rank factorization of alpha
    if not R == "full":
        U = np.abs(np.random.randn(M, R))
        lu = len(U)
        u_mu = np.repeat(0, M)
        V = np.abs(np.random.randn(K, R))
        lv = len(V)
        v_mu = np.repeat(0, K)

        # TODO
        lambda_ = 1

        x = np.hstack((U.flatten(), V.flatten(), u_mu, v_mu))
        x = randn(len(x)) / 100

        par_uv = {'getu': range(0, lu), \
                'getv': range(lu, lu + lv), \
                'getumean': range(lu + lv, lu + lv + M), \
                'getvmean': range(lu + lv + M, len(x)), \
                'M': M, \
                'K': K, \
                'R': R, \
                'D': D, \
                'lambda': lambda_}
        
        par_uv_w2 = np.zeros((M, K))


    cost = []  # for storing the lower bounds

    #
    # The main loop
    #
    for iter_ in range(int(iter_max)):
        pass


# TODO: remove later, just for testing, to see if this shit runs
if __name__ == "__main__":
    Y_1 = np.array([[-1, 1], [-2, 2]])
    Y_2 = np.array([[10, 11, 12], [20, 22, 24], [30, 33, 36], [40, 44, 48]])
    Y = [Y_1, Y_2]
    gfa(Y, K=8)
