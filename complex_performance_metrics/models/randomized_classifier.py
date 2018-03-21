import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from complex_performance_metrics.solvers.binary import \
    coco_hmean_cov, coco_qmean_dp, coco_err_dp, fraco_fmeasure_kld
from complex_performance_metrics.solvers.multiclass import \
    coco_qmean_nae, fraco_microF1_cov
import complex_performance_metrics.utils as utils


"""
This module contains a class for implementing a randomized classifier
"""


class RandomizedClassifier:
    '''
    Implements a randomized classifier

    Attributes:
        weights (list of floats): Weights on individual plug-in classifier
        classifiers (list of objects): List of plug-in classifiers
        protected_present (bool): Does the dataset contain a protected attribute?
        num_class (int): Number of classes
        loss_name (string): Name of loss function
        cons_name (string): Name of constraint function
        opt_name (string): Name of solver ('coco' or 'fraco')
    '''

    def __init__(self, loss_name, cons_name, protected_present=False, num_class=2):
        """
        Initialize class

        Args:
            loss_name (string): Name of loss function
            cons_name (string): Name of constraint function
            protected_present (bool): Does the dataset contain a protected attribute? (default: False)
            num_class (int): Number of classes (default: 2)
        """

        self.weights = []
        self.classifiers = list()
        self.protected_present = protected_present
        self.num_class = num_class
        self.loss_name = loss_name
        self.cons_name = cons_name
        self.opt_name = None

    def append(self, w, plugin):
        """
        Append (weight, classifier) to randomized classifier

        Args:
            w (float): weight on classifier
            plugin (string): Name of constraint function
            protected_present (bool): Does the dataset contain a protected attribute? (default: False)
            num_class (int): Number of classes (default: 2)
        """

        if type(w) == list:
            self.weights.append(w)
            self.classifiers.append(plugin)
        else:
            self.weights += [w]
            self.classifiers += [plugin]

    def normalize_weights(self):
        """
        Normalize weights of randomized classifier
        """

        tot = sum(self.weights)
        if tot > 0:
            self.weights = [x*1.0/tot for x in self.weights]

    def fit_(self, x, y, eps, eta, num_outer_iter, num_inner_iter, cpe_model=None, z=None):
        """
        Fit a randomized classifier using solver opt_name that optimizes loss_name s.t. cons_name

        Args:
           x (array-like, dtype = float, shape=(m,d)): Features
           y (array-like, dtype = int, shape=(m,)): Labels {0,...,m-1}
           eps (float): Constraint function tolerance
           eta (float): Step-size for gradient-ascent solver
           num_outer_iter (int): Number of outer iterations in solver (gradient ascent)
           num_inner_iter (int): Number of inner iterations in solver (Frank-Wolfe)
           cpe_model (sklearn estimator): A model with a predict_proba() function (default: None)
           z (array-like, dtype = int, shape=(m,)): Protected attribute {0,..M} (default: None)
        """
        # If cpe_model not specified, fit a LogReg model
        if cpe_model is None:
            cpe_model = LogisticRegressionCV()
            cpe_model.fit(x, y)

        # Get module for performance measure / constraint, raise exception if solver not available
        module_name = self.opt_name + '_' + self.loss_name + '_' + self.cons_name
        if module_name not in globals():
            raise KeyError('No solver found for optimizing ' + self.loss_name + ' under ' + self.cons_name + 'constraint')
        module = globals()[module_name]

        # Check if there is a protected attribute
        if not self.protected_present:
            module.fit(x, y, self, cpe_model, eps, eta, num_outer_iter, num_inner_iter)
        else:
            self.protected_present = True
            module.fit(x, y, z, self, cpe_model, eps, eta, num_outer_iter, num_inner_iter)

    def evaluate_conf(self, x_ts, y_ts, z_ts=None, use_stored_prob=False):
        """
        Calculate confusion matrix

        Args:
            x_ts (array-like, dtype = float, shape = (m,d)): Test features
            y_ts (array-like, dtype = int, shape = (m,)): Test labels {0,...,m-1}
            z_ts (array-like, dtype = int, shape = (m,)): Test protected attribute {0,..M} (default: None)
            use_stored_prob (bool): Use probabilities computed from previous calls

        Returns:
            conf (array-like, dtype = float, shape = (n,n)): Confusion matrix
        """
        if not self.protected_present:
            C = np.zeros((self.num_class, self.num_class))
            for t in range(len(self.weights)):
                C += self.weights[t] * self.classifiers[t].evaluate_conf(
                    x_ts, y_ts, use_stored_prob=use_stored_prob)
            return C
        else:
            M = len(np.unique(z_ts))
            C = np.zeros((self.num_class, self.num_class))
            CC = np.zeros((M, self.num_class, self.num_class))
            for t in range(len(self.weights)):
                C_, CC_ = self.classifiers[t].evaluate_conf(
                    x_ts, y_ts, z_ts, use_stored_prob=use_stored_prob)
                C += self.weights[t] * C_
                for j in range(M):
                    CC[j, :, :] += self.weights[t] * CC_[j, :, :]
            return C, CC

    def evaluate_loss(self, x_ts, y_ts, z_ts=None):
        """
        Calculate loss function

        Args:
            x_ts (array-like, dtype = float, shape = (m,d)): Test features
            y_ts (array-like, dtype = int, shape = (m,)): Test labels {0,...,m-1}
            z_ts (array-like, dtype = int, shape = (m,)): Test protected attribute {0,..M} (default: None)

        Returns:
            loss (float): Loss function value
        """
        if not self.protected_present:
            C = self.evaluate_conf(x_ts, y_ts)
        else:
            C, _ = self.evaluate_conf(x_ts, y_ts, z_ts)
        return utils.evaluate_metric(self.loss_name, C)
