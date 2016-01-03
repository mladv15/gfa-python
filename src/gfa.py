"""
GFA (Group Factor Analysis)
This is a Python implementation of the file ./R/CCAGFA.R in the R package CCAGFA
"""

from __future__ import division, print_function
import numpy as np
import scipy as sp
import scipy.special
import scipy.linalg
import scipy.optimize
import math

DEBUG = True


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
        if not np.all(np.abs(np.mean(Y_m, axis=0)) < 1e-7):
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
        digammaa_ard = sp.special.psi(a_ard)
    tau = np.repeat(init_tau, M)    # The mean noise precisions
    a_tau = alpha_0t + N*D/2        # The parameters of the Gamma distribution
    b_tau = np.zeros(M)             #   for the noise precisions
    digammaa_tau = sp.special.psi(a_tau)  # Constants needed for computing the lower bound
    lgammaa_tau = -np.sum(np.vectorize(math.lgamma)(a_tau))
    lb_pt_const = -M*np.vectorize(math.lgamma)(alpha_0t) + M*alpha_0t*np.log(beta_0t)

    # Alpha needs to be initialized to match the data scale
    for m in range(M):
        alpha[m, :] = K*D[m]/(datavar[m]-1/tau[m])

    # The projections
    # No need to initialize projections randomly ,since their updating
    # step is the first one; just define the variables here
    #low_mem = True
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

    # Rotation parameters (full rank only)
    if(rotate):
        Rot = np.diag(np.ones(K))       # The rotation matrix R (in ICML11 paper)
        RotInv = np.diag(np.ones(K))    # Its inverse
        r = np.array(Rot).flatten()     # Vectorizd version of R, will be passed to optimization function

        # parameter dict for the optimization function
        # scipy.optimize takes these optional parameters as a tuple and passes them to the objective function 
        # but store them as dict first for easier modification
        par_dict = {'K': K, 'D': D, 'Ds': Ds, 'N': N, 'WW': WW, 'ZZ': ZZ, 'M': M}

    
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
        x = np.random.randn(len(x)) / 100

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
        
        # Check if some components need to be removed
        # remove columns which have most elements approaching 0
        # np.where() returns a tuple
        (keep,) = np.where(np.power(Z, 2).mean(axis=0) > 1e-7)  # column indices to keep
        id_ = np.ones(K)
        if len(keep) != K and dropK:
            K = len(keep)
            if K == 0:
                raise ValueError("All latent factors in Z are 0, shut down all components, no structure found in the data")
            id_ = np.ones(K)
            # in R, when selecting only one column from the matrix, the result is defaulted to
            # a normal (row) array. Since we're indexing with an array (`keep`), the Python default
            # is to return a column vector, so no need for a drop argument.
            Z = Z[:, keep]
            covZ = covZ[keep, keep]
            ZZ = ZZ[keep, keep]
            for m in range(M):
                W[m] = W[m][:, keep]
                if not low_mem:
                    covW[m] = covW[m][keep, keep]
                WW[m] = WW[m][keep, keep]

            alpha = alpha[:, keep]
            logalpha = logalpha[:, keep]

            if R != "full":
                V = V[keep, :]
                v_mu = v_mu[keep]
                # TODO: update x, lv and par
            else:
                b_ard = np.ones((M, K))
            if rotate:
                par_dict['K'] = K
        # endif len(keep) != K and dropK

        #
        # Update the projections
        #
        lb_qw = np.empty(M)  # Computes also the determinant of covW needed for the lower bound
        for m in range(M):
            # Efficient and robust way of computing
            # solve(diag(alpha) + tau * ZZ^T)
            tmp = 1/np.sqrt(alpha[m, :])
            # Cholesky decomposition
            # R package uses upper triangular part, as does scipy (but NOT numpy)
            diag_tau = np.diag(np.tile(tau, K)[:K])
            cho_before = np.outer(tmp, tmp) * ZZ + diag_tau
            cho = sp.linalg.cholesky(cho_before)
            det = -2*np.sum(np.log(np.diag(cho))) - np.sum(np.log(alpha[m, :])) - K*np.log(tau[m])
            lb_qw[m] = det
            if not low_mem:
                # chol2inv calculates the inverse of the matrix whose Choleski decomposition was given.
                # Python doesn't have this function, so I'll just take the inverse of the matrix itself
                # without going through its Cholesky decomposition
                covW[m] = 1/tau[m] * np.outer(tmp, tmp) * np.linalg.inv(cho_before)
                W[m] = np.dot(Y[m].T, Z).dot(covW[m]) * tau[m]
                WW[m] = np.dot(W[m].T, W[m]) + covW[m]*D[m]
            else:
                covW = 1/tau[m] * np.outer(tmp, tmp) * np.linalg.inv(cho_before)
                W[m] = np.dot(Y[m].T, Z).dot(covW) * tau[m]
                WW[m] = np.dot(W[m].T, W[m]) + covW*D[m]

        # 
        # Update the latent variables
        #
        
        # Efficient and robust way of computing
        # solve(diag(1,K) + tau * WW^t)
        covZ = np.diag(np.ones(K))
        for m in range(M):
            covZ = covZ + tau[m]*WW[m]
        cho = sp.linalg.cholesky(covZ, lower=False)
        covZ = np.linalg.inv(covZ)
        det = -2*np.sum(np.log(np.diag(cho)))
        lb_qx = det

        Z = Z*0
        for m in range(M):
            Z = Z + Y[m].dot(W[m])*tau[m]
        Z = Z.dot(covZ)
        ZZ = np.dot(Z.T, Z) + N*covZ

        #
        # Optimization of the rotation (only start after the first
        # iteration)
        #

        if R=="full" and rotate and iter_ > 0:
            #Update the parameter list for the optimizer
            par_dict["WW"] = WW
            par_dict["ZZ"] = ZZ

            # par <- list(K=K,D=D,Ds=Ds,N=N,WW=WW,ZZ=ZZ,M=M)
            par = tuple([par_dict[key] for key in ['K', 'D', 'Ds', 'N', 'WW', 'ZZ', 'M']])

            # Always start from the identity matrix, i.e. no rotation
            r = np.diag(np.ones(K)).flatten()
            if opt_method == "BFGS":
                r_opt = sp.optimize.minimize(fun=E, x0=r, args=par, method='BFGS', jac=gradE,
                                             options={'maxiter': opt_iter})  # no reltol in SciPy
            if opt_method == "L-BFGS":
                r_opt = sp.optimize.minimize(fun=E, x0=r, args=par, method='L-BFGS-B', jac=gradE,
                                             options={'maxiter': opt_iter})  # factr deprecated

            # print(r_opt)
            if not r_opt.success:
                # sometimes work, indicating that the loss function E and the gradient gradE are correct?
                # mostly doesn't work though because the code is not complete yet.
                print("\n=============================================================")
                print("Failure in optimizing the rotation. Turning the rotation off.")
                print("=============================================================\n")
                rotate = False
            else:
                # Update the parameters involved in the rotation:
                Rot = r_opt.x.reshape(K, K)
                U, d, V = np.linalg.svd(Rot)
                det = np.sum(np.log(d))
                RotInv = np.dot( V*np.outer(id_, 1/d), U.T )

                Z = np.dot(Z, RotInv.T)
                covZ = np.dot(RotInv.dot(covZ), RotInv.T)
                ZZ = np.dot(Z.T, Z) + N*covZ

                lb_qx = lb_qx - 2*det

                for m in range(M):
                    if not low_mem:
                        W[m] = W[m].dot(Rot)
                        covW[m] = np.dot(Rot, covW[m].T).dot(Rot)
                        WW[m] = np.dot(W[m].T, W[m]) + covW[m]*D[m]
                    else:
                        # covW[m] is not stored, so it needs to be computed before rotation
                        covW = (WW[m] - np.dot(W[m].T, W[m]))/D[m]
                        W[m] = W[m].dot(Rot)
                        covW = np.dot(Rot.T, covW).dot(Rot)
                        WW[m] = np.dot(W[m].T, W[m]) + covW*D[m]

                    lb_qw[m] = lb_qw[m] + 2*det

        # moar stuff to come ...

        # TODO: change calculation of lower bound
        if verbose == 2:
            print("Iteration: %d/ cost: 1337/ K: %d" % (iter_, K))
        """
        # Convergence if the relative change in cost is small enough
        if iter_ > 0:
            diff = cost[iter_] - cost[iter_-1]
            if abs(diff)/abs(cost[iter_]) < iter_crit or iter_ == iter_max:
                break
        """
        # TODO: remove this later, this is just here to prevent us from doing the default `iter_max`=10e5 loops
        if iter_ > 10:
            break

    if DEBUG:
        pass
        """
        print("Z")
        print(np.array(Z))
        print("covZ")
        print(covZ)
        print("ZZ")
        print(ZZ)

        print("W")
        print(np.array(W))
        print("covW")
        print(covW)
        print("WW")
        print(WW)
        """


def E(r, K, D, Ds, N, WW, ZZ, M):
    """
    Evaluates the (negative) cost function valule wrt the transformation
    matrix R used in the generic optimization routine

    `r` is the flattened array of the rotation matrix R (see ICML11 paper)
    """
    R = np.array(r).reshape(K, K)
    (U, d, V) = np.linalg.svd(R)

    tmp = U*np.outer(np.ones(K), 1/d)
    val = -np.sum(ZZ*np.dot(tmp, tmp.T))/2
    val = val + (Ds-N)*np.sum(np.log(d))
    for m in range(M):
        val = val - D[m]*np.sum( np.log( (R*(WW[m].dot(R))).mean(axis=0) ) )
    return -val


def gradE(r, K, D, Ds, N, WW, ZZ, M):
    """
    Evaluates the (negative) gradient of the cost of the function E()
    """
    R = np.array(r).reshape(K, K)
    U, d, V = np.linalg.svd(R)
    Rinv = np.dot( V*np.outer(np.ones(K), 1/(d**2)), U.T )
    gr_tmp = np.dot( U*np.outer(np.ones(K), 1/(d**2)), U.T ).dot(ZZ) \
           + np.diag(np.ones(K)*(Ds-N))
    gr = np.dot(gr_tmp, Rinv.T).flatten()

    tmp1 = WW[0].dot(R)
    tmp2 = 1/(R*tmp1).mean(axis=0)
    tmp1 = D[0] * (tmp1*np.outer(np.ones(K), tmp2)).flatten()
    gr = gr - tmp1
    for m in range(1, M):
        tmp1 = WW[m].dot(R)
        tmp2 = 1/(R*tmp1).mean(axis=0)
        tmp1 = D[m] * (tmp1*np.outer(np.ones(K), tmp2)).flatten()
        gr = gr - tmp1
    return -gr


# TODO: remove later, just for testing, to see if this shit runs
if __name__ == "__main__":
    Y_1 = np.array([[1, 2], [-1, -2]])
    Y_2 = np.array([[10, 11, 12], [-10, -11, -12]])
    Y = [Y_1, Y_2]
    gfa(Y, K=8)
