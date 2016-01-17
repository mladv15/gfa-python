"""
Python implementation of W projections (Figure 3 in the paper)
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.spatial import distance
from src.gfa import gfa
from src.gfa import gfa_experiments

# Dimensions
N = 100  # num samples
M = 3  # num groups
D = [10, 10, 10]  # data dimensions for each group, D = [D_1, ..., D_M]
K = 7  # num latent factors


def main():
    W, Z, Y = generate_data()

    # Plot true latent components
    for k in range(K):
        plt.subplot(K, 1, k+1)
        plt.scatter(range(N), Z[:, k], facecolors='none')
    plt.suptitle("True latent components")
    """
    # Remove later, but could be interesting to plot the actual observations
    # Group 1 is 15-dimensional, so I'll just
    # plot observations of group 2 (5 dimensional)
    plt.figure()
    for d_m in range(D[1]):
        plt.subplot(D[1], 1, d_m+1)
        plt.scatter(range(N), Y[1][:, d_m], facecolors='none')
    plt.suptitle("Observations of group 2 (5-dimensional)")
    """

    model = gfa_experiments(Y, K=8, Nrep=10, rotate=False, verbose=1)

    # plot estaimated latent components
    plt.figure()
    for k in range(model['K']):
        plt.subplot(model['K'], 1, k+1)
        plt.scatter(range(N), model['Z'][:, k], facecolors='none')
    plt.suptitle("Estimated active latent components")

    # reorder estimated latent variables to correspond to the true order

    W_conc = [None]*K
    Wmodel_conc = [None]*K
    for k in range(K):
        W_conc[k] = np.concatenate([W[m][:, k] for m in range(M)])
        Wmodel_conc[k] = np.concatenate([model['W'][m][:, k] for m in range(M)])
    # order_map[x] = y means that estimated latent variable number x
    # corresponds to the true latent variable number y
    order_map = [None]*K
    for k in range(K):
        w_k_model = Wmodel_conc[k]
        similarities = [distance.cosine(np.abs(w_k_model), np.abs(W_conc[i])) for i in range(K)]
        most_sim_idx = np.argmin(similarities)
        print(similarities)
        order_map[k] = most_sim_idx
    print(order_map)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(np.abs(W_conc), cmap=plt.cm.gray_r, interpolation='none')
    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(Wmodel_conc), cmap=plt.cm.gray_r, interpolation='none')
    plt.title("W_model concatenate")
    plt.show()

    # plot true matrix projections
    plot_W(W, "True")

    plt.figure()
    # plot estimated matrix projections
    plot_W(model['W'], "Estimated")

    plt.show()


def plot_W(W, title):
    M = len(W)
    for m, W_m in enumerate(W):
        plt.subplot(M, 1, m+1)
        plt.imshow(np.abs(W_m), cmap=plt.cm.gray_r, interpolation='none')
        # turn off axes ticks
        plt.tick_params(
            axis='both',        # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom='off',       # ticks along the bottom edge are off
            top='off',          # ticks along the top edge are off
            labelbottom='off')  # labels along the bottom edge are off
        # plt.colorbar()
    plt.suptitle(title)


def generate_data():
    # Data generated according to GFA model assumption
    # as seen in Section 3 Model, in the paper

    # Latent samples
    Z = np.random.randn(N, K)

    # Precisions
    tau_prior = gamma(1e-14, scale=1.0/(1e-14))  # scale = 1/rate
    tau = tau_prior.rvs(M)  # noise precision for each group
    # this gave me a list of zeros, but we have to divide by tau later -> division by 0
    # just set to 1 for now
    tau = np.array([1]*M)

    o = 1e6  # inactive weights: high precision -> low variance -> inactive (close to 0)
    l = 1    # active weights: low precision -> low variance -> inactive (close to 0)
    alpha = np.zeros((M, K))  # component precisions for each group
    alpha[0, :] = [l, o, l, l, o, o, l]  # component precisions for group 1
    alpha[1, :] = [l, l, o, l, o, l, o]  # component precisions for group 2
    alpha[2, :] = [l, l, l, o, l, o, o]  # component precisions for group 3

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
        # Y[m] = Y[m][:Ntrain, :]

    # split latent samples into training and test sets
    # Ztest = Z[(Ntrain+1):, :]
    # Z = Z[:Ntrain, :]

    return W, Z, Y


if __name__ == "__main__":
    main()
