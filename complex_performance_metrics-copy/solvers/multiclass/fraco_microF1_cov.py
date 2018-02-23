import numpy as np
from copy import copy
from complex_performance_metrics.utils import microF1, cov
from complex_performance_metrics.models.plugin import MulticlassPluginClassifier


# Implementation of FRACO algorithm
#    for optimizing micro-F1 s.t. cov <= \eps


def coco_microF1_cov(x, y, classifier, cpe_model, thresh, eps, eta, max_iter):
    # COCO to optimize linearized micro-F1 at thresh s.t. cov <= eps
    if eps == 1:
        max_iter = 1
        lamda = 0
    else:
        lamda = 0.01

    num_class = classifier.num_class

    plugin = MulticlassPluginClassifier(cpe_model, num_class=num_class)
    # Store predictions for 'x'
    plugin.predict(x)

    C = np.zeros((num_class, num_class))
    for t in range(max_iter):
        W = np.zeros((num_class, num_class))
        for j in range(1, num_class):
            W[j, j] = 2

        W[0, 0] = 2 * thresh
        W[1:, 0] = thresh + lamda
        W[0, 1:] = thresh

        plugin.set_cost_matrix(W, is_cost=False)
        C_hat = plugin.evaluate_conf(x, y, use_stored_prob=True)

        classifier.append(1.0, copy(plugin))
        C = C + C_hat / max_iter

        coverage = cov(C_hat)
        lamda += eta * 1.0 / np.sqrt(t + 1) * (coverage - eps)
        # Projection
        if (lamda < 0):
            lamda = 0

    classifier.normalize_weights()

    return C


def fit(x, y, classifier, cpe_model, eps, eta, max_outer_iter, max_inner_iter=None):
    # FRACO, outer bisection method
    lwr = 0
    upr = 1

    while upr - lwr > 0.01:
        thresh = (lwr + upr) / 2.0

        C = coco_microF1_cov(
            x, y, classifier, cpe_model, thresh, eps, eta, max_outer_iter)

        fm = microF1(C)

        if fm < thresh:
            upr = thresh
        else:
            lwr = thresh

    return classifier
