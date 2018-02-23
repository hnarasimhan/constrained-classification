import numpy as np
from copy import copy
from complex_performance_metrics.models.plugin import BinaryPluginClassifier


# Implementation of COCO algorithm
#    for optimizing H-mean s.t. Coverage <= \eps


def frank_wolfe(x, y, classifier, cpe_model, alpha, max_inner_iter):
    # Frank-wolfe algorithm to minimize Lagrangian

    p = y.mean()
    pos_coef = 1.0 / p
    neg_coef = 1.0 / (1 - p)

    C = np.zeros((2, 2))

    plugin = BinaryPluginClassifier(cpe_model)
    # Store predictions for 'x'
    plugin.predict(x)

    norm_const = 1.0

    for i in range(max_inner_iter):
        pos_coef0 = pos_coef - alpha
        neg_coef0 = neg_coef + alpha

        thresh0 = neg_coef0 / (pos_coef0 + neg_coef0)
        plugin.set_thresh(thresh0)

        C_hat = plugin.evaluate_conf(x, y, use_stored_prob=True)
        C = (1 - 2.0 / (i + 2)) * C + 2.0 / (i + 2) * C_hat

        if i == 0:
            classifier.append(1.0, copy(plugin))
        else:
            norm_const *= 1 - 2.0 / (i + 2)
            classifier.append(2.0 / (i + 2) / norm_const, copy(plugin))

        tpr = C[1, 1] * 1.0 / (C[1, 0] + C[1, 1])
        tnr = C[0, 0] * 1.0 / (C[0, 0] + C[0, 1])

        pos_coef = 2 * tnr * tnr * 1.0 / (tpr + tnr) / (tpr + tnr) / p
        neg_coef = 2 * tpr * tpr * 1.0 / (tpr + tnr) / (tpr + tnr) / (1 - p)

    classifier.weights[-max_inner_iter:-1] = [x * norm_const for x in classifier.weights[-max_inner_iter:-1]]
    classifier.weights[-1] *= norm_const

    return C, classifier


def fit(x, y, classifier, cpe_model, eps, eta, max_outer_iter, max_inner_iter):
    # COCO algorithm, outer subgradient ascent
    if eps == 1:
        alpha = 0
        max_outer_iter = 1
    else:
        alpha = 0.01

    for t in range(max_outer_iter):
        C,_ = frank_wolfe(x, y, classifier, cpe_model, alpha, max_inner_iter)

        cov = C[0, 1] + C[1, 1]
        alpha = alpha + eta * 1.0 / np.sqrt(t + 1) * (cov - eps)

        # Projection
        if alpha < 0:
            alpha = 0

    classifier.normalize_weights()

    return classifier
