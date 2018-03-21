import numpy as np
from copy import copy
from complex_performance_metrics.models.plugin import MulticlassPluginClassifier
from complex_performance_metrics.utils import nae


"""
This module implements the COCO algorithm for: 
    minimizing the (multiclass) Q-mean loss subject to (multiclass) Normalized Absolute Error <= epsilon
"""


def frank_wolfe(x, y, classifier, cpe_model, alpha, num_inner_iter):
    """
    Inner Frank-Wolfe optimization in COCO
    Perform Lagrangian optimization over confusion matrices for Lagrange multiplier alpha

    Args:
      x (array-like, dtype = float, shape = (m,d)): Features
      y (array-like, dtype = int, shape = (m,)): Labels {0,...,m-1}
      classifier (RandomizedClassifier):
        A randomized classifier to which additional base classifiers are to be added
      cpe_model (sklearn estimator): A model with a predict_proba() function (default: None)
      alpha (float): Lagrange multiplier
      num_inner_iter (int): Number of solver iterations

    Returns:
      classifier (RandomizedClassifier): Solution classifier for inner maximization
    """
    num_class = classifier.num_class

    p = np.zeros((num_class,))
    for i in range(num_class):
        p[i] = (y == i).mean()

    # Intialize gain matrix
    W = np.eye(num_class, num_class)
    for i in range(num_class):
         W[i, i] = 1.0 / p[i]

    # Create binary plug-in with separate thresholds for protected attribute values
    plugin = MulticlassPluginClassifier(cpe_model, num_class=num_class)
    plugin.set_cost_matrix(W, is_cost=False)
    C = plugin.evaluate_conf(x, y, use_stored_prob=True)

    # Initialize constants/normalization terms
    norm_costs_const = 0.5 / (1 - np.min(p))
    norm_weights_const = 1.0

    # Frank-Wolfe iterations
    for i in range(num_inner_iter):
        # Compute gain matrix from objective gradient,
        #    and construct optimal classifier for cost-sensitive learning step
        W = np.zeros((num_class, num_class))
        for j in range(num_class):
            for k in range(num_class):
                if j == k:
                    W[j, j] = 2 * (1 - C[j, j] / p[j]) / p[j] / num_class
                W[j, k] -= alpha * 2 * (C[:,k].sum() - p[k]) * norm_costs_const
        plugin.set_cost_matrix(W, is_cost=False)

        # Update confusion matrix iterate
        C_hat = plugin.evaluate_conf(x, y, use_stored_prob=True)
        C = (1 - 2.0 / (i + 2)) * C + 2.0 / (i + 2) * C_hat

        # Append weight and copy of plug-in classifier to randomized classifier
        if i == 0:
            classifier.append(1.0, copy(plugin))
        else:
            norm_weights_const *= 1 - 2.0 / (i + 2)
            classifier.append(2.0 / (i + 2) / norm_weights_const, copy(plugin))

    # Normalize classifier weights to sum up to 1
    classifier.weights[-num_inner_iter:-1] = [x * norm_weights_const for x in classifier.weights[-num_inner_iter:-1]]
    classifier.weights[-1] *= norm_weights_const

    return C, classifier


def fit(x, y, classifier, cpe_model, eps, eta, num_outer_iter, num_inner_iter):
    """
       Outer optimization in COCO
       Run gradient ascent over Lagrange multipliers alpha

       Args:
         x (array-like, dtype = float, shape= (m,d)): Features
         y (array-like, dtype = int, shape = (m,)): Labels {0,...,m-1}
         classifier (RandomizedClassifier):
                A randomized classifier to which additional base classifiers are to be added
         cpe_model (sklearn estimator): A model with a predict_proba() function (default: None)
         eps (float): Constraint function tolerance
         eta (float): Step-size for gradient-ascent solver
         num_outer_iter (int): Number of outer iterations in solver (gradient ascent)
         num_inner_iter (int): Number of inner iterations in solver (Frank-Wolfe)

    Returns:
      classifier (RandomizedClassifier): Final classifier
    """
    # If the problem has no constraints, set alpha to 0 and run outer solver only for one iteration
    if eps == 1:
        alpha = 0.0
        num_outer_iter = 1
    else: # else initialize Lagrange multiplier
        alpha = 0.01

    # Gradient ascent iterations
    for t in range(num_outer_iter):
        # Find confusion matrix that maximizes the Lagrangian at alpha
        C, _ = frank_wolfe(x, y, classifier, cpe_model, alpha, num_inner_iter)
        er = nae(C)

        # Gradient update to alpha
        alpha += eta * 1.0 / np.sqrt(t + 1) * (er - eps)

        # Projection step
        if alpha < 0:
            alpha = 0

    # Normalize classifier weights to sum up to 1
    classifier.normalize_weights()
    return classifier
