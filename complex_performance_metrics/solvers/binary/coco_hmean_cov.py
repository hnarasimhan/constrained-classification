import numpy as np
from copy import copy
from complex_performance_metrics.models.plugin import BinaryPluginClassifier


"""
This module implements the COCO algorithm for: 
    minimizing the H-mean loss subject to Coverage <= epsilon
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
    p = y.mean()
    pos_coef = 1.0 / p
    neg_coef = 1.0 / (1 - p)

    C = np.zeros((2, 2))

    # Create binary plug-in
    plugin = BinaryPluginClassifier(cpe_model)
    plugin.predict(x) # Stores predictions for x within object

    # Initialize normalization term
    norm_const = 1.0

    # Frank-Wolfe iterations
    for i in range(num_inner_iter):
        # Compute costs from objective gradient
        pos_coef0 = pos_coef - alpha
        neg_coef0 = neg_coef + alpha

        # Optimal classifier for cost-sensitive learning step
        thresh0 = neg_coef0 / (pos_coef0 + neg_coef0)
        plugin.set_thresh(thresh0)

        # Update confusion matrix iterate
        C_hat = plugin.evaluate_conf(x, y, use_stored_prob=True) # Use stored probability
        C = (1 - 2.0 / (i + 2)) * C + 2.0 / (i + 2) * C_hat

        # Append weight and copy of plug-in classifier to randomized classifier
        if i == 0:
            classifier.append(1.0, copy(plugin))
        else:
            norm_const *= 1 - 2.0 / (i + 2)
            classifier.append(2.0 / (i + 2) / norm_const, copy(plugin))

        # Maintain stats for computing cost in next iteration
        tpr = C[1, 1] * 1.0 / (C[1, 0] + C[1, 1])
        tnr = C[0, 0] * 1.0 / (C[0, 0] + C[0, 1])

        pos_coef = 2 * tnr * tnr * 1.0 / (tpr + tnr) / (tpr + tnr) / p
        neg_coef = 2 * tpr * tpr * 1.0 / (tpr + tnr) / (tpr + tnr) / (1 - p)

    # Normalize classifier weights to sum up to 1
    classifier.weights[-num_inner_iter:-1] = [x * norm_const for x in classifier.weights[-num_inner_iter:-1]]
    classifier.weights[-1] *= norm_const

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
        alpha = 0
        num_outer_iter = 1
    else: # else initialize Lagrange multiplier
        alpha = 0.01

    # Gradient ascent iterations
    for t in range(num_outer_iter):
        # Find confusion matrix that maximizes the Lagrangian at alpha
        C,_ = frank_wolfe(x, y, classifier, cpe_model, alpha, num_inner_iter)

        # Gradient update to alpha based on gradient at optimal C
        cov = C[0, 1] + C[1, 1]
        alpha = alpha + eta * 1.0 / np.sqrt(t + 1) * (cov - eps)

        # Projection step
        if alpha < 0:
            alpha = 0

    # Normalize classifier weights to sum up to 1
    classifier.normalize_weights()

    return classifier
