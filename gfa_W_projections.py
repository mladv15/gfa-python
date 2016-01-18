"""
Python implementation of W projections (Figure 3 in the paper)
"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.spatial import distance
from sklearn.decomposition import FactorAnalysis

from collections import Counter

from src.gfa import gfa
from src.gfa import gfa_experiments

# Dimensions
N = 100  # num samples
M = 3  # num groups
D = [10, 10, 10]  # data dimensions for each group, D = [D_1, ..., D_M]
K = 7  # num latent factors


def main():
    W, Z, Y = generate_data()
    # for FA
    Y_hstack = np.hstack([Y[m] for m in range(M)])

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

    # GFA
    model = gfa_experiments(Y, K=8, Nrep=10, rotate=False, verbose=1)
    # FA
    fa = FactorAnalysis(K)
    fa.fit(Y_hstack)

    # plot estaimated latent components from GFA
    plt.figure()
    for k in range(model['K']):
        plt.subplot(model['K'], 1, k+1)
        plt.scatter(range(N), model['Z'][:, k], facecolors='none')
    plt.suptitle("GFA: estimated active latent components")

    # plot estaimated latent components from FA
    plt.figure()
    Z_fa = fa.transform(Y_hstack)
    for k in range(model['K']):
        plt.subplot(model['K'], 1, k+1)
        plt.scatter(range(N), Z_fa[:, k], facecolors='none')
    plt.suptitle("FA: estimated active latent components")

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
        w_k = W_conc[k]
        similarities = [distance.cosine(np.abs(w_k), np.abs(Wmodel_conc[i])) for i in range(K)]
        most_sim_idx = np.argmin(similarities)
        order_map[k] = most_sim_idx

    ###################
    # TODO: get reordering from Z, not W
    order_map_Z = [None]*K
    for k in range(K):
        z_k = Z[:, k]
        similarities = [distance.cosine(z_k, model['Z'][:, i]) for i in range(K)]
        print(similarities)
        most_sim_idx = np.argmin(similarities)
        order_map_Z[k] = most_sim_idx
    print(order_map_Z)
    ###################

    Wmodel_ordered = np.array(Wmodel_conc)[order_map].T
    plt.figure()
    plt.imshow(np.abs(Wmodel_ordered), cmap=plt.cm.gray_r, interpolation='none')

    # plot true matrix projections
    plt.figure()
    plot_grouped_W(W, "True")

    plt.figure()
    # plot estimated matrix projections
    #plot_grouped_W(model['W'], "Estimated")

    plt.show()


def plot_grouped_W(W, title):
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


def get_latent_order(Z, Zmodel):
    assert(np.all(Z.shape == Zmodel.shape))
    K = Z.shape[1]
    # z_similarities[k] = list of tuples, sorted from most similar modeled latent factor index to the least similar
    # z_similarities[k] = [(z_model_idx, similarity)]
    z_similarities = [None]*K
    # z_closest_idx[k] = list of estimated latent factor indices that are closest to the true z_k, sorted from closest to furthest
    z_closest_idx = [None]*K
    # z_closest_sim[k] = list of distances for corresponding indices in z_closest_idx
    z_closest_sim = [None]*K
    for k in range(K):
        z_k = Z[:, k]
        similarities = [distance.cosine(z_k, Zmodel[:, i]) for i in range(K)]
        sim_argsort = np.argsort(similarities)
        z_similarities[k] = zip(sim_argsort, similarities[sim_argsort])
    # order_map[k] = i means that true latent variable number k
    # corresponds to the estimated latent variable number i
    order_map = [None]*K
    for trial in range(K):
        # the closest index for this trial, but do not include z_k's closest if z_k has already been assigned a model latent factor
        closest_idx_trial = [None]*K
        closest_sim_trial = [None]*K
        for k, mapped in enumerate(order_map):
            if mapped is None:
                closest_idx_trial[k] = z_similarities[k][trial][0]
                closest_sim_trial[k] = z_similarities[k][trial][1]
        # closest_idx_trial = [z_similarities[k][trial][0] for k in range(K)]
        # closest_sim_trial = [z_similarities[k][trial][1] for k in range(K)]
        idx_counter = Counter(closest_idx_trial)
        idx_counter.pop(None, None)
        for k, closest_idx in enumerate(closest_idx_trial):
            # set z_k's closest if not already set and closest_idx is unique for this trial
            if closest_idx is not None and idx_counter[closest_idx] == 1:
                order_map[k] = closest_idx



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
