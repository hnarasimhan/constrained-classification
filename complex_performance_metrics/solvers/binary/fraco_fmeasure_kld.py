import numpy as np
from copy import copy
from complex_performance_metrics.models.plugin import BinaryPluginClassifier
from complex_performance_metrics.utils import kld


"""
This module implements the FRACO algorithm for: 
    minimizing the F-measure loss subject to KL-divergence Error <= epsilon
"""


def coco_fmeasure_kld(x, y, classifier, cpe_model, gamma, eps, eta, num_outer_iter, num_inner_iter):
    """
        Inner COCO optimization in FRACO
        Solve convex optimization:
            minimize f(C) - gamma f'(C) over all C for which KLD(C) <= eps for gammaold gamma

        Args:
          x (array-like, dtype = float, shape = (m,d)): Features
          y (array-like, dtype = int, shape = (m,)): Labels {0,...,m-1}
          classifier (RandomizedClassifier):
            A randomized classifier to which additional base classifiers are to be added
          cpe_model (sklearn estimator): A model with a predict_proba() function (default: None)
          gamma (float): gammaold on fractional-convex loss
          eps (float): Constraint function tolerance
          eta (float): Step-size for gradient-ascent solver
          num_outer_iter (int): Number of gradient-ascent iterations
          num_inner_iter (int): Number of Frank-Wolfe iterations

        Returns:
          classifier (RandomizedClassifier): Solution classifier for convex optimization
          obj (float): Objective value at solution classifier, i.e. f(C^*) - gamma f'(C^*)
        """
    # If the problem has no constraints, set lamda to 0 and run outer
    #           and inner solvers only for one iteration
    if eps == 1:
        num_outer_iter = 1
        num_inner_iter = 1
        lamda = 0
    else: # Else initialize Lagrange multiplier
        lamda = 0.01

    p = y.mean()

    plugin = BinaryPluginClassifier(cpe_model)
    plugin.predict(x)  # Stores class probability estimates for 'x'

    obj = 0
    # Gradient ascent iterations
    for t in range(num_outer_iter):
        C = np.zeros((2, 2))
        norm_const = 1.0

        # Inner Frank-Wolfe solver:
        #   Find confusion matrix that maximizes the Lagrangian at lamda
        for i in range(num_inner_iter):
            # Compute costs from objective gradient
            wt_on_neg = gamma - lamda * p / (p + C[0, 1] - C[1, 0])\
                        + lamda * (1 - p) / (1 - p - C[0, 1] + C[1, 0])
            wt_on_pos = 2 - gamma + lamda * p / (p + C[0, 1] - C[1, 0])\
                        - lamda * (1 - p) / (1 - p - C[0, 1] + C[1, 0])

            # Optimal classifier for cost-sensitive learning step
            gamma0 = wt_on_neg * 1.0 / (wt_on_pos + wt_on_neg)
            plugin.set_thresh(gamma0)

            # Update confusion matrix iterate
            C_hat = plugin.evaluate_conf(x, y, use_stored_prob=True)
            C = (1 - 2.0 / (i + 2)) * C + 2.0 / (i + 2) * C_hat

            # Append weight and copy of plug-in classifier to randomized classifier
            if i == 0:
                classifier.append(1.0, copy(plugin))
            else:
                norm_const *= 1 - 2.0 / (i + 2)
                classifier.append(2.0 / (i + 2) / norm_const, copy(plugin))

        # Normalize classifier weights across the last
        #       num_inner_iter Frank-Wolfe iterations to 1
        classifier.weights[-num_inner_iter:-1] = \
            [v * norm_const for v in classifier.weights[-num_inner_iter:-1]]
        classifier.weights[-1] *= norm_const

        # Gradient update to lamda based on gradient at C
        qloss = kld(C)
        lamda = lamda + eta * (qloss - eps)

        # Projection step
        if lamda < 0:
            lamda = 0

        obj = 2.0 * (1 - gamma) * C[1, 1] - gamma * (C[1, 0] + C[0, 1])

    # Normalize classifier weights across gradient ascent iterations to sum up to 1
    classifier.normalize_weights()

    return classifier, obj


def fit(x, y, classifier, cpe_model, eps, eta, num_outer_iter, num_inner_iter=1):
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
         num_outer_iter (int): Number of gradient ascent iterations in COCO
         num_inner_iter (int): Number of Frank-Wolfe iterations in COCO

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

        # Run COCO to compute: fm = min_C f(C) - gamma f'(C) s.t. KLD(C) <= eps
        classifier, obj = coco_fmeasure_kld(
            x, y, classifier, cpe_model, gamma, eps, eta, num_outer_iter, num_inner_iter)

        # Update upper or lower bound based on whether or not fm < gamma
        if obj < 0:
            upr = gamma
        else:
            lwr = gamma

    return classifier
