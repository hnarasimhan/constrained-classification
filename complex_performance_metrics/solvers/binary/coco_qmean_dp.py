import numpy as np
from copy import copy
from complex_performance_metrics.models.plugin import BinaryPluginClassifier


"""
This module implements the COCO algorithm for: 
    minimizing the Q-mean loss subject to Demographic Parity <= epsilon
"""


def frank_wolfe(x, y, z, classifier, cpe_model, alpha, num_inner_iter):
    """
    Inner Frank-Wolfe optimization in COCO
    Perform Lagrangian optimization over confusion matrices for Lagrange multiplier alpha

    Args:
      x (array-like, dtype = float, shape = (m,d)): Features
      y (array-like, dtype = int, shape = (m,)): Labels {0,...,m-1}
      z (array-like, dtype = int, shape = (m,)): Protected attribute {0,..M}
      classifier (RandomizedClassifier):
            A randomized classifier to which additional base classifiers are to be added
      cpe_model (sklearn estimator): A model with a predict_proba() function (default: None)
      alpha (array-like, dtype = float, shape = (M,)): Lagrange multiplier
      num_inner_iter (int): Number of solver iterations

    Returns:
      classifier (RandomizedClassifier): Solution classifier for inner maximization
    """
    p = y.mean()

    M = len(np.unique(z))
    pp = np.zeros((M, 1))
    for i in range(M):
        pp[i] = y[z == i].mean()

    nn = np.zeros((M, 1))
    for i in range(M):
        nn[i] = len(z[z == i])

    # Create binary plug-in with separate thresholds for protected attribute values
    plugin = BinaryPluginClassifier(cpe_model, protected_present=True)
    plugin.set_thresh([0.5] * M)
    C, CC = plugin.evaluate_conf(x, y, z, use_stored_prob=True)

    # Initialize constants/normalization terms
    norm_const = 1.0
    psquare = p * p
    one_minus_psquare = (1 - p) * (1 - p)

    # Frank-Wolfe iterations
    for i in range(num_inner_iter):
        # Compute costs from objective gradient,
        #    and construct optimal classifier for cost-sensitive learning step
        thresh = [0.0] * M  # needs to be a list
        for j in range(M):
            pos_coeff = pp[j] / psquare * C[1, 0] \
                        - (alpha[j] * (1 + 1.0 / M) - np.mean(alpha))
            neg_coeff = pp[j] / one_minus_psquare * C[0, 1] \
                        + (alpha[j] * (1 + 1.0 / M) - np.mean(alpha))
            thresh[j] = neg_coeff * 1.0 / (pos_coeff + neg_coeff)
        plugin.set_thresh(thresh)

        # Update confusion matrix iterate
        C_hat, CC_hat = plugin.evaluate_conf(x, y, z, use_stored_prob=True)  # use stored probabilities
        C = (1 - 2.0 / (i + 2)) * C + 2.0 / (i + 2) * C_hat
        for j in range(M):
            CC[j, :, :] = (1 - 2.0 / (i + 2)) * CC[j, :, :] + 2.0 / (i + 2) * CC_hat[j, :, :]

        # Append weight and copy of plug-in classifier to randomized classifier
        if i == 0:
            classifier.append(1.0, copy(plugin))
        else:
            norm_const *= 1 - 2.0 / (i + 2)
            classifier.append(2.0 / (i + 2) / norm_const, copy(plugin))

    # Normalize classifier weights to sum up to 1
    classifier.weights[-num_inner_iter:-1] = [x * norm_const for x in classifier.weights[-num_inner_iter:-1]]
    classifier.weights[-1] *= norm_const

    return C, CC, classifier


def fit(x, y, z, classifier, cpe_model, eps, eta, num_outer_iter, num_inner_iter):
    """
       Outer optimization in COCO
       Run gradient ascent over Lagrange multipliers alpha

       Args:
         x (array-like, dtype = float, shape= (m,d)): Features
         y (array-like, dtype = int, shape = (m,)): Labels {0,...,m-1}
         z (array-like, dtype = int, shape = (m,)): Protected attribute {0,..M}
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
    M = len(np.unique(z))
    s = np.ones((M,))
    alpha = np.zeros((M,))

    # If the problem has no constraints, run outer solver only for one iteration
    if eps == 1:
        num_outer_iter = 1

    # Gradient ascent iterations
    for t in range(num_outer_iter):
        # Find confusion matrix that maximizes the Lagrangian at alpha
        C, CC, _ = frank_wolfe(x, y, z, classifier, cpe_model, alpha * s, num_inner_iter)

        # Compute gradient at optimal confusion matrix C
        C_mean = np.zeros((2, 2))
        for j in range(M):
            C_mean += CC[j, :, :].reshape((2, 2)) * 1.0 / M

        jstar = np.argmax(np.abs(CC[:, 0, 1] + CC[:, 1, 1] - C_mean[0, 1] - C_mean[1, 1]) - eps)

        for j in range(M):
            s[j] = np.sign(CC[jstar, 0, 1] + CC[jstar, 1, 1] - C_mean[0, 1] - C_mean[1, 1])

        # Gradient update to alpha
        for j in range(M):
            alpha[j] = alpha[j] + eta * 1.0 / np.sqrt(t + 1) \
                       * (np.abs(CC[jstar, 0, 1] + CC[jstar, 1, 1] - C_mean[0, 1] - C_mean[1, 1]) - eps)

            # Projection step
            if alpha[j] < 0:
                alpha[j] = 0

    # Normalize classifier weights to sum up to 1
    classifier.normalize_weights()

    return classifier
