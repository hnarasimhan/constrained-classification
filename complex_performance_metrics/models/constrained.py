from randomized_classifier import RandomizedClassifier
import complex_performance_metrics.utils as utils


"""
This module contains classes for optimizing complex loss functions under convex constraints, 
using the COCO and FRACO algorithms in Narasimhan (2016)
"""


class ConstrainedClassifier(RandomizedClassifier):
    """
    Base class for optimizing complex loss functions under constraint
    """

    def fit(self, x, y, eps, eta, num_outer_iter, num_inner_iter=1, cpe_model=None, z=None):
        """
        Fit a randomized classifier

        Args:
           x (array_like, dtype = float, shape=(m,d)): Features
           y (array_like, dtype = int, shape=(m,)): Labels {0,...,m-1}
           eps (float): Constraint function tolerance
           eta (float): Step-size for gradient-ascent solver
           num_outer_iter (int): Number of outer iterations in solver (gradient ascent)
           num_inner_iter (int): Number of inner iterations in solver (Frank-Wolfe)
           cpe_model (sklearn estimator): A model with a predict_proba() function (default: None)
           z (array_like, dtype = int, shape=(m,)): Protected attribute {0,..M} (default: None)
        """
        RandomizedClassifier.fit_(self, x, y, eps, eta, num_outer_iter, num_inner_iter, cpe_model, z)

    def evaluate_cons(self, x_ts, y_ts, z_ts=None):
        """
        Calculate constraint function

        Args:
           x_ts (array_like, dtype = float, shape=(m,d)): Test features
           y_ts (array_like, dtype = int, shape=(m,)): Test labels {0,...,m-1}
           z_ts (array_like, dtype = int, shape=(m,)): Test protected attribute {0,..M} (default: None)

        Returns:
           cons (float): Constraint function value
        """
        if not self.protected_present:
            C = self.evaluate_conf(x_ts, y_ts)
        else:
            _, C = self.evaluate_conf(x_ts, y_ts, z_ts)
        return utils.evaluate_metric(self.cons_name, C)


class COCOClassifier(ConstrainedClassifier):
    """
    Learns a randomized classifier using the COCO algorithm for constrained convex losses
    Derived from base class ConstrainedClassifier

    Attributes:
        opt_name (string): Solver name
    """

    def __init__(self, loss_name, cons_name, protected_present=False, num_class=2):
        """
        Initialize class

        Args:
            loss_name (string): Name of loss function
            cons_name (string): Name of constraint function
            protected_present (bool): Does the dataset contain a protected attribute? (default: False)
            num_class (int): Number of classes (default: 2)
        """
        RandomizedClassifier.__init__(self, loss_name, cons_name, protected_present, num_class)
        self.opt_name = 'coco'


class FRACOClassifier(ConstrainedClassifier):
    """
    Learns a randomized classifier using the FRACO algorithm for constrained fractional-convex losses
    Derived from base class ConstrainedClassifier

    Attributes:
        opt_name (string): Solver name
    """

    def __init__(self, loss_name, cons_name, protected_present=False, num_class=2):
        """
        Initialize class

        Args:
            loss_name (string): Name of loss function
            cons_name (string): Name of constraint function
            protected_present (bool): Does the dataset contain a protected attribute? (default: False)
            num_class (int): Number of classes (default: 2)
        """
        RandomizedClassifier.__init__(self, loss_name, cons_name, protected_present, num_class)
        self.opt_name = 'fraco'
