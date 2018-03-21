import numpy as np
from copy import copy
from complex_performance_metrics.models.plugin import BinaryPluginClassifier
from complex_performance_metrics.utils import kld

# Implementation of FRACO algorithm
#    for optimizing F-measure s.t. KL-divergence <= \eps


def coco_fmeasure_kld(x, y, classifier, cpe_model, thresh, eps, eta, num_outer_iter, num_inner_iter):
    # COCO to optimize linearized fmeasure at thresh s.t. kld <= eps
    if eps == 1:
        num_outer_iter = 1
        num_inner_iter = 1
        lamda = 0
    else:
        lamda = 0.01

    p = y.mean()

    plugin = BinaryPluginClassifier(cpe_model)
    # Store predictions for 'x'
    plugin.predict(x)

    obj = 0

    for t in range(num_outer_iter):
        C = np.zeros((2, 2))
        norm_const = 1.0

        # Inner Frank-Wolfe solver
        for i in range(num_inner_iter):
            wt_on_neg = thresh - lamda * p / (p + C[0, 1] - C[1, 0]) + lamda * (1 - p) / (1 - p - C[0, 1] + C[1, 0])
            wt_on_pos = 2 - thresh + lamda * p / (p + C[0, 1] - C[1, 0]) - lamda * (1 - p) / (1 - p - C[0, 1] + C[1, 0])

            thresh0 = wt_on_neg * 1.0 / (wt_on_pos + wt_on_neg)
            plugin.set_thresh(thresh0)

            C_hat = plugin.evaluate_conf(x, y, use_stored_prob=True)
            C = (1 - 2.0 / (i + 2)) * C + 2.0 / (i + 2) * C_hat

            if i == 0:
                classifier.append(1.0, copy(plugin))
            else:
                norm_const *= 1 - 2.0 / (i + 2)
                classifier.append(2.0 / (i + 2) / norm_const, copy(plugin))

        classifier.weights[-num_inner_iter:-1] = [v * norm_const for v in classifier.weights[-num_inner_iter:-1]]
        classifier.weights[-1] *= norm_const

        obj = 2.0 * (1 - thresh) * C[1, 1] - thresh * (C[1, 0] + C[0, 1])
        qloss = kld(C)

        lamda = lamda + eta * (qloss - eps)
        # Projection
        if lamda < 0:
            lamda = 0

    classifier.normalize_weights()

    return classifier, obj


def fit(x, y, classifier, cpe_model, eps, eta, num_outer_iter, num_inner_iter=1):
    # FRACO, outer bisection method
    lwr = 0
    upr = 1

    while upr - lwr > 0.01:
        thresh = (lwr + upr) / 2.0

        classifier, obj = coco_fmeasure_kld(
            x, y, classifier, cpe_model, thresh, eps, eta, num_outer_iter, num_inner_iter)

        if obj < 0:
            upr = thresh
        else:
            lwr = thresh

    return classifier
