"""
Python implementation of the file ./demo/CCAGFAExperiment.R in the R package CCAGFA
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from src.gfa import gfa_experiments

# Dimensions
Ntrain = Ntest = 100
N = Ntrain + Ntest  # num samples
M = 2  # num groups
D = [15, 7]  # data dimensions for each group, D = [D_1, ..., D_M]
K = 4  # num latent factors


def main():
    (Z, Y), (Ztest, Ytest) = generate_data()
    K = 4

    # TODO: train and test GFA
    # model = gfa(Y, K=8)

    # Plot true latent components
    for k in range(K):
        plt.subplot(4, 1, k+1)
        plt.scatter(range(Ntrain), Z[:, k], facecolors='none')

    # Remove later, but could be interesting to plot the actual observations
    # Group 1 is 15-dimensional, so I'll just
    # plot observations of group 2 (5 dimensional)
    plt.figure()
    for d_m in range(D[1]):
        plt.subplot(D[1], 1, d_m+1)
        plt.scatter(range(Ntrain), Y[1][:, d_m], facecolors='none')
    plt.suptitle("Observations of group 2 (5-dimensional)")
    plt.show()


def generate_data():
    # Latent samples
    Z = np.empty((N, K))
    Z[:, 0] = np.sin(np.linspace(1, N, N)/(N/20))  # first factor is sin
    Z[:, 1] = np.cos(np.linspace(1, N, N)/(N/20))  # second factor is cos
    Z[:, 2] = np.random.randn(N)  # third factor is N(0, 1)
    # fourth factor is just a straight line
    Z[:Ntrain, 3] = np.linspace(1, Ntrain, Ntrain)/Ntrain - 0.5  # training samples
    Z[Ntrain:, 3] = np.linspace(1, Ntest, Ntest)/Ntest - 0.5  # test samples

    # Precisions
    tau = [3, 6]  # noise precision for each group
    alpha = np.empty((M, K))  # component precisions for each group
    alpha[0, :] = [1, 1, 1e6, 1]  # component precisions for group 1
    alpha[1, :] = [1, 1, 1e6, 1]  # component precisions for group 2

    # Observations
    #   Y    : List of M data matrices. Y[m] is a matrix with
    #          N rows (samples) and D_m columns (features). The
    #          samples need to be co-occurring.
    #          NOTE: All of these should be centered, so that the mean
    #                of each feature is zero
    #          NOTE: The algorithm is roughly invariant to the scale
    #                of the data, but extreme values should be avoided.
    #                Data with roughly unit variance or similar scale
    #                is recommended.
    Y = [None]*M
    Ytest = [None]*M
    # Factor loadings
    # W[m] is actually the transposed (D_m x K) matrix W^(m).T
    # from equation (1) in the paper
    W = [None]*M

    # for each group m
    for m in range(M):
        W[m] = np.empty((D[m], K))
        # for each latent factor k
        for k in range(K):
            # set factor loadings of factor k for group m
            # each factor loading W[m][d_m, k] is sampled independently
            # from N(0, 1/alpha[m, k])
            W[m][:, k] = np.random.randn(D[m])/np.sqrt(alpha[m, k])
        # observations for group m
        # each group has its own noise, sampled indep. from N(0, 1/tau[m])
        epsilon_m = np.random.randn(N, D[m])/np.sqrt(tau[m])
        Y[m] = np.dot(Z, W[m].T) + epsilon_m
        # split observations into training and test sets
        Ytest[m] = Y[m][(Ntrain+1):, :]
        Y[m] = Y[m][:Ntrain, :]

    # split latent samples into training and test sets
    Ztest = Z[(Ntrain+1):, :]
    Z = Z[:Ntrain, :]

    return (Z, Y), (Ztest, Ytest)


if __name__ == "__main__":
    main()
