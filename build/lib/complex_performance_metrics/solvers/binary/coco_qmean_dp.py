import numpy as np
from copy import copy
from complex_performance_metrics.models.plugin import BinaryPluginClassifier


def frank_wolfe(x, y, z, classifier, cpe_model, alpha, max_inner_iter):
    p = y.mean()

    M = len(np.unique(z))
    pp = np.zeros((M, 1))
    for i in range(M):
        pp[i] = y[z == i].mean()

    nn = np.zeros((M, 1))
    for i in range(M):
        nn[i] = len(z[z == i])

    plugin = BinaryPluginClassifier(cpe_model, protected_present=True)
    plugin.set_thresh([0.5] * M)
    C, CC = plugin.evaluate_conf(x, y, z, use_stored_prob=True)

    norm_const = 1.0
    psquare = p * p
    one_minus_psquare = (1 - p) * (1 - p)

    for i in range(max_inner_iter):
        thresh = [0.0] * M  # needs to be a list
        for j in range(M):
            pos_coeff = pp[j] / psquare * C[1, 0] \
                        - (alpha[j] * (1 + 1.0 / M) - np.mean(alpha))
            neg_coeff = pp[j] / one_minus_psquare * C[0, 1] \
                        + (alpha[j] * (1 + 1.0 / M) - np.mean(alpha))
            thresh[j] = neg_coeff * 1.0 / (pos_coeff + neg_coeff)
        plugin.set_thresh(thresh)

        C_hat, CC_hat = plugin.evaluate_conf(x, y, z, use_stored_prob=True)
        C = (1 - 2.0 / (i + 2)) * C + 2.0 / (i + 2) * C_hat
        for j in range(M):
            CC[j, :, :] = (1 - 2.0 / (i + 2)) * CC[j, :, :] + 2.0 / (i + 2) * CC_hat[j, :, :]

        if i == 0:
            classifier.append(1.0, copy(plugin))
        else:
            norm_const *= 1 - 2.0 / (i + 2)
            classifier.append(2.0 / (i + 2) / norm_const, copy(plugin))

    classifier.weights[-max_inner_iter:-1] = [x * norm_const for x in classifier.weights[-max_inner_iter:-1]]
    classifier.weights[-1] *= norm_const

    return C, CC, classifier


def fit(x, y, z, classifier, cpe_model, eps, eta, max_outer_iter, max_inner_iter):
    M = len(np.unique(z))
    s = np.ones((M,))
    alpha = np.zeros((M,))

    if eps == 1:
        max_outer_iter = 1

    for t in range(max_outer_iter):
        C, CC, _ = frank_wolfe(x, y, z, classifier, cpe_model, alpha * s, max_inner_iter)

        C_mean = np.zeros((2, 2))
        for j in range(M):
            C_mean += CC[j, :, :].reshape((2, 2)) * 1.0 / M

        jstar = np.argmax(np.abs(CC[:, 0, 1] + CC[:, 1, 1] - C_mean[0, 1] - C_mean[1, 1]) - eps)

        for j in range(M):
            s[j] = np.sign(CC[jstar, 0, 1] + CC[jstar, 1, 1] - C_mean[0, 1] - C_mean[1, 1])

        for j in range(M):
            alpha[j] = alpha[j] + eta * 1.0 / np.sqrt(t + 1) \
                       * (np.abs(CC[jstar, 0, 1] + CC[jstar, 1, 1] - C_mean[0, 1] - C_mean[1, 1]) - eps)

            # Projection
            if alpha[j] < 0:
                alpha[j] = 0

    classifier.normalize_weights()

    return classifier

