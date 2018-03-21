import numpy as np
from copy import copy
from complex_performance_metrics.utils import microF1, cov
from complex_performance_metrics.models.plugin import MulticlassPluginClassifier


"""
This module implements the FRACO algorithm for: 
    minimizing the (multiclass) micro F1 loss subject to Coverage <= epsilon
"""


def coco_microF1_cov(x, y, classifier, cpe_model, gamma, eps, eta, num_iter):
    """
    Inner COCO optimization in FRACO
    Solve convex optimization:
        minimize f(C) - gamma f'(C) over all C for which Cov(C) <= eps for threshold gamma

    Args:
      x (array-like, dtype = float, shape = (m,d)): Features
      y (array-like, dtype = int, shape = (m,)): Labels {0,...,m-1}
      classifier (RandomizedClassifier):
        A randomized classifier to which additional base classifiers are to be added
      cpe_model (sklearn estimator): A model with a predict_proba() function (default: None)
      gamma (float): Threshold on fractional-convex loss
      eps (float): Constraint function tolerance
      eta (float): Step-size for gradient-ascent solver
      num_iter (int): Number of gradient-ascent iterations

    Returns:
      classifier (RandomizedClassifier): Solution classifier for convex optimization
    """
    # If the problem has no constraints, set lamda to 0 and run outer solver only for one iteration
    if eps == 1:
        num_iter = 1
        lamda = 0
    else: # Else initialize Lagrange multiplier
        lamda = 0.01

    num_class = classifier.num_class

    plugin = MulticlassPluginClassifier(cpe_model, num_class=num_class)
    plugin.predict(x)  # Stores class probability estimates for 'x'

    # Gradient ascent iterations
    C = np.zeros((num_class, num_class))
    for t in range(num_iter):
        # Find confusion matrix that maximizes the Lagrangian at lamda
        #  (has closed-form solution, so no need for inner Frank-Wolfe)
        W = np.zeros((num_class, num_class))
        for j in range(1, num_class):
            W[j, j] = 2

        W[0, 0] = 2 * gamma
        W[1:, 0] = gamma + lamda
        W[0, 1:] = gamma

        plugin.set_cost_matrix(W, is_cost=False)
        C_hat = plugin.evaluate_conf(x, y, use_stored_prob=True)

        # Add plug-in classifier to randomized classifier with a weight of 1.0
        classifier.append(1.0, copy(plugin))

        # Maintain average confusion matrix
        C = C + C_hat / num_iter

        # Gradient update to lamda based on gradient at C_hat
        coverage = cov(C_hat)
        lamda += eta * 1.0 / np.sqrt(t + 1) * (coverage - eps)

        # Projection step
        if (lamda < 0):
            lamda = 0

    # Normalize classifier weights to sum up to 1
    classifier.normalize_weights()

    return C


def fit(x, y, classifier, cpe_model, eps, eta, num_iter, dummy_iter=None):
    """
       Outer optimization in FRACO
       Run bisection method to perform binary search over threshold gamma

       Args:
         x (array-like, dtype = float, shape= (m,d)): Features
         y (array-like, dtype = int, shape = (m,)): Labels {0,...,m-1}
         classifier (RandomizedClassifier):
                A randomized classifier to which additional base classifiers are to be added
         cpe_model (sklearn estimator): A model with a predict_proba() function (default: None)
         eps (float): Constraint function tolerance
         eta (float): Step-size for gradient-ascent solver
         num_iter (int): Number of gradient ascent iterations in COCO
         dummy_iter: Dummy parameter to confirm to the syntax of the fit function in other solvers

    Returns:
      classifier (RandomizedClassifier): Final classifier
    """
    # Maintain lower and upper bounds on optimal loss
    lwr = 0
    upr = 1

    # Bisection iterations: loop until the difference between lower and upper bounds is > 0.01
    while upr - lwr > 0.01:
        # Choose threshold as average of lower and upper bounds
        gamma = (lwr + upr) / 2.0

        # Run COCO to compute: fm = min_C f(C) - gamma f'(C) s.t. Cov(C) <= eps
        C = coco_microF1_cov(x, y, classifier, cpe_model, gamma, eps, eta, num_iter)
        fm = 1.0 - microF1(C)

        # Update upper or lower bound based on whether or not fm < gamma
        if fm < gamma:
            upr = gamma
        else:
            lwr = gamma

    return classifier
