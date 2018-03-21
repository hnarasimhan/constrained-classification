import numpy as np
from copy import copy
from complex_performance_metrics.models.plugin import BinaryPluginClassifier


"""
This module implements the COCO algorithm for: 
    minimizing the 0-1 Classification Error subject to Demographic Parity <= epsilon
"""


def opt_classifier(x, y, z, classifier, cpe_model, alpha):
    """
    Inner optimization in COCO
    Perform Lagrangian optimization over confusion matrices for Lagrange multiplier alpha

    Args:
      x (array-like, dtype = float, shape = (m,d)): Features
      y (array-like, dtype = int, shape = (m,)): Labels {0,...,m-1}
      z (array-like, dtype = int, shape = (m,)): Protected attribute {0,..M}
      classifier (RandomizedClassifier):
            A randomized classifier to which additional base classifiers are to be added
      cpe_model (sklearn estimator): A model with a predict_proba() function (default: None)
      alpha (array-like, dtype = float, shape = (M,)): Lagrange multiplier

    Returns:
      C (array-like, dtype = float, shape = (n,n)): Confusion matrix for solution classifier
      CC (array-like, dtype = float, shape = (M,n,n)):
            Confusion matrix for solution classifier, for each protected group 1, ..., M
      classifier (RandomizedClassifier): Solution classifier for inner maximization
    """
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


def fit(x, y, z, classifier, cpe_model, eps, eta, num_iter, dummy_iter=None):
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
      dummy_iter: Dummy parameter to confirm to the syntax of the fit function in other solvers

    Returns:
      classifier (RandomizedClassifier): Final classifier
    """
    M = len(np.unique(z))
    s = np.ones((M,))
    alpha = np.zeros((M,))

    # Gradient ascent iterations
    for t in range(num_iter):
        # Find confusion matrix that maximizes the Lagrangian at alpha
        C, CC, _ = opt_classifier(x, y, z, classifier, cpe_model, alpha * s)

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
