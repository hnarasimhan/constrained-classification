import numpy as np
from copy import copy
from complex_performance_metrics.models.plugin import BinaryPluginClassifier


def opt_classifier(x, y, z, classifier, cpe_model, alpha):
    M = len(np.unique(z))
    pp = np.zeros((M, 1))
    for i in range(M):
        pp[i] = y[z == i].mean()

    plugin = BinaryPluginClassifier(cpe_model, protected_present=True)
    thresh = [0.0] * M  # needs to be a list
    for j in range(M):
        pos_coeff = pp[j] - (alpha[j] * (1 + 1.0 / M) - np.mean(alpha))
        neg_coeff = pp[j] + (alpha[j] * (1 + 1.0 / M) - np.mean(alpha))
        thresh[j] = neg_coeff * 1.0 / (pos_coeff + neg_coeff)
    plugin.set_thresh(thresh)

    C, CC = plugin.evaluate_conf(x, y, z, use_stored_prob=True)
    classifier.append(1.0, copy(plugin))

    return C, CC, classifier


def fit(x, y, z, classifier, cpe_model, eps, eta, max_outer_iter, max_inner_iter=None):
    M = len(np.unique(z))
    s = np.ones((M,))
    alpha = np.zeros((M,))

    if eps == 1:
        max_outer_iter = 1

    # Gradient ascent
    for t in range(max_outer_iter):
        C, CC, _ = opt_classifier(x, y, z, classifier, cpe_model, alpha * s)

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

