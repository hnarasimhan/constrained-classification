from randomized_classifier import RandomizedClassifier
import complex_performance_metrics.solvers as solvers


"""
This module contains classes for optimizing complex loss functions without constraints, 
using the Frank-Wolfe and Bisection algorithms in Narasimhan et al. (2015)
"""


class FrankWolfeClassifier(RandomizedClassifier):
    """
    Learns a randomized classifier using the Frank-Wolfe algorithm for unconstrained convex losses
    Derived from base class RandomizedClassifier
    """

    def __init__(self, loss_name, protected_present=False, num_class=2):
        """
        Initialize class

        Args:
            loss_name (string): Name of performance measure
            protected_present (bool): Does the dataset contain a protected attribute? (default: False)
            num_class (int): Number of classes (default: 2)
        """
        if loss_name not in solvers.PERF_CONS_MAP:
            raise KeyError("No solver found for optimizing " + loss_name)
        RandomizedClassifier.__init__(self, loss_name, solvers.PERF_CONS_MAP[loss_name], protected_present, num_class)
        self.opt_name = 'coco' # Use COCO solver with no constraints

    def fit(self, x, y, num_iter, cpe_model=None, z=None):
        """
        Fit a randomized classifier using the Frank-Wolfe algorithm that optimizes loss_name

        Args:
            x (array-like, dtype = float, shape=(m,d)): Features
            y (array-like, dtype = int, shape=(m,)): Labels {0,...,m-1}
            num_iter (int): Number of iterations
            cpe_model (sklearn estimator): A model with a predict_proba() function (default: None)
            z (array-like, dtype = int, shape=(m,)): Protected attribute {0,..M} (default: None)
        """
        # Invoke the COCO fit() with eps = 1, eta = 1, num_outer_iter = num_iter, num_inner_iter = 1
        RandomizedClassifier.fit_(self, x, y, 1, 1, num_iter, 1, cpe_model, z)


class BisectionClassifier(RandomizedClassifier):
    """
    Learns a randomized classifier using the Bisection algorithm for unconstrained fractional-convex losses
    Derived from base class RandomizedClassifier
    """

    def __init__(self, loss_name, protected_present=False, num_class=2):
        """
        Initialize class

        Args:
            loss_name (string): Name of performance measure
            protected_present (bool): Does the dataset contain a protected attribute? (default: False)
            num_class (int): Number of classes (default: 2)
        """
        if loss_name not in solvers.PERF_CONS_MAP:
            raise KeyError("No solver found for optimizing " + loss_name)
        RandomizedClassifier.__init__(self, loss_name, solvers.PERF_CONS_MAP[loss_name], protected_present, num_class)
        self.opt_name = 'fraco' # Use FRACO solver

    def fit(self, x, y, cpe_model=None, z=None):
        """
        Fit a randomized classifier using the Frank-Wolfe algorithm that optimizes loss_name

        Args:
            x (array-like, dtype = float, shape=(m,d)): Features
            y (array-like, dtype = int, shape=(m,)): Labels {0,...,m-1}
            cpe_model (sklearn estimator): A model with a predict_proba(x) function (default: None)
            z (array-like, dtype = int, shape=(m,)): Protected attribute {0,..M} (default: None)
        """
        # Invoke the FRACO fit() with eps = 1 (no constraint), eta = 1, num_outer_iter = 1, num_inner_iter = 1
        RandomizedClassifier.fit_(self, x, y, 1, 1, 1, 1, cpe_model, z)
