import numpy as np
from copy import copy
from complex_performance_metrics.models.plugin import MulticlassPluginClassifier
from complex_performance_metrics.utils import nae
from complex_performance_metrics.utils import qmean

def frank_wolfe(x, y, classifier, cpe_model, alpha, max_inner_iter):
    num_class = classifier.num_class

    p = np.zeros((num_class,))
    for i in range(num_class):
        p[i] = (y == i).mean()

    W = np.eye(num_class, num_class)
    for i in range(num_class):
         W[i, i] = 1.0 / p[i]

    plugin = MulticlassPluginClassifier(cpe_model, num_class=num_class)
    plugin.set_cost_matrix(W, is_cost=False)
    C = plugin.evaluate_conf(x, y, use_stored_prob=True)

    norm_costs_const = 0.5 / (1 - np.min(p))
    norm_weights_const = 1.0

    for i in range(max_inner_iter):
        W = np.zeros((num_class, num_class))
        for j in range(num_class):
            # for k in range(num_class):
            #     if j != k:
            #         W[j, k] = 2 * (C[j, :].sum() - C[j, j]) / p[j] / p[j] \
            #                   - alpha * np.sign(C[:, j].sum() - C[j, :].sum()) * norm_costs_const \
            #                   + alpha * np.sign(C[:, k].sum() - C[k, :].sum()) * norm_costs_const
            #     # else:
            for k in range(num_class):
                if j == k:
                    W[j, j] = 2 * (1 - C[j, j] / p[j]) / p[j] / num_class
                W[j, k] -= alpha * 2 * (C[:,k].sum() - p[k]) * norm_costs_const
        plugin.set_cost_matrix(W, is_cost=False)

        C_hat = plugin.evaluate_conf(x, y, use_stored_prob=True)
        C = (1 - 2.0 / (i + 2)) * C + 2.0 / (i + 2) * C_hat

        if i == 0:
            classifier.append(1.0, copy(plugin))
        else:
            norm_weights_const *= 1 - 2.0 / (i + 2)
            classifier.append(2.0 / (i + 2) / norm_weights_const, copy(plugin))

    # print qmean(C), nae(C)

    classifier.weights[-max_inner_iter:-1] = [x * norm_weights_const for x in classifier.weights[-max_inner_iter:-1]]
    classifier.weights[-1] *= norm_weights_const

    return C, classifier


def fit(x, y, classifier, cpe_model, eps, eta, max_outer_iter, max_inner_iter):
    if eps == 1:
        alpha = 0.0
        max_outer_iter = 1
    else:
        alpha = 0.01

    for t in range(max_outer_iter):
        C, _ = frank_wolfe(x, y, classifier, cpe_model, alpha, max_inner_iter)
        er = nae(C)
        # print t, er

        alpha += eta * 1.0 / np.sqrt(t + 1) * (er - eps)
        # Projection
        if alpha < 0:
            alpha = 0

        # print 'outer', t, qmean(C), er, alpha, max_outer_iter

    classifier.normalize_weights()
    return classifier
