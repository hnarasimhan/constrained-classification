import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import complex_performance_metrics.utils as utils


"""
This module contains classes for binary and multiclass plug-in classifiers 
"""


class PluginClassifier:
    """
    Base class for plug-in classifier

    Attributes:
        cpe_model (sklearn estimator): A model with a predict_proba() function
    """

    def fit_cpe(self, x, y):
        """
        Fit class probability estimation model

        Args:
            x (array-like, dtype=float, shape=(m,d)): Instances
            y (array-like, dtype=int, shape=(m,)): Labels
        """
        self.cpe_model = LogisticRegressionCV(solver='liblinear')
        self.cpe_model.fit(x, y)

    def set_cpe(self, cpe_model):
        """
        Set class probability estimation model

        Args:
            cpe_model (sklearn estimator): A model with a predict_proba() function
        """
        self.cpe_model = cpe_model


class BinaryPluginClassifier(PluginClassifier):
    """
    Binary plug-in classifier with separate thresholds for protected attribute values
    Derived from base class PluginClassifier

    Attributes:
        cpe_model (sklearn estimator): A model with a predict_proba() function
        t0 (float or list of M floats): Threshold for group 0 or list of thresholds for M groups
        t1 (float): Threshold for group 1 (default = None)
        yprob (array-like, dtype = float, shape = (m,)): Stored estimated probabilities for m instances
        protected_present: Does the dataset contain a protected attribute (default = False)
    """

    def __init__(self, cpe_model=None, t0=0.5, t1=None, protected_present=False):
        """
        Initialize class

        Attributes:
            cpe_model (sklearn estimator): A model with a predict_proba() function
            t0 (float or list of M floats): Threshold for group 0 or list of thresholds for M groups
            t1 (float): Threshold for group 1 (default = None)
                (specify either thresholds (t0, t1) or a threshold list t0)
            protected_present: Does the dataset contain a protected attribute (default = False)
        """
        self.cpe_model = cpe_model
        self.t0 = t0
        self.t1 = t1
        self.yprob = None
        self.protected_present = protected_present

    def set_thresh(self, t0=0.5, t1=None):
        """
        Set thresholds for plug-in classifier

        Attributes:
            t0 (float or list of M floats): Threshold for group 0 or list of thresholds for M groups
            t1 (float): Threshold for group 1 (default = None)
                (specify either thresholds (t0, t1) or a threshold list t0)
        """
        self.t0 = t0
        self.t1 = t1

    def predict(self, x_ts, z_ts=None, use_stored_prob=False):
        """
        Predict labels using plug-in classifier

        Args:
            x_ts (array-like, dtype = float, shape = (m,d)): Test features

        Returns:
            ypred (array-like, dtype = float, shape = (m,)): Predicted labels for m data points
        """
        if (not use_stored_prob) or (self.yprob is None) :
            self.yprob = self.cpe_model.predict_proba(x_ts)[:,1]
        ypred = np.zeros((len(self.yprob),))
        if not self.protected_present:
            # No protected attribute
            ypred = 1.0 * (self.yprob >= self.t0)
        elif type(self.t0) == list:
            # Multiple protected groups with thresholds in list thresh
            for i in range(len(self.t0)):
                ypred[z_ts == i] = (self.yprob[z_ts == i] > self.t0[i]) * 1.0
        else:
            # Two protected groups with thresholds t0 and t1
            ypred[z_ts == 0] = (self.yprob[z_ts == 0] > self.t0) * 1.0
            ypred[z_ts == 1] = (self.yprob[z_ts == 1] > self.t1) * 1.0
        return ypred

    def predict_proba(self, x_ts):
        """
        Predict probabilities using cpe_model, and store them in yprob

        Args:
            x_ts (array-like, dtype = float, shape = (m,d)): Test features

        Returns:
            yprob (array-like, dtype = float, shape = (m,)): Predicted probabilities for m data points
        """
        self.yprob = self.cpe_model.predict_proba(x_ts)[:,1]
        return self.yprob

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
        ypred = self.predict(x_ts, z_ts, use_stored_prob)
        C = metrics.confusion_matrix(y_ts, ypred) * 1.0 / x_ts.shape[0]
        if not self.protected_present:
            return C
        else:
            M = len(np.unique(z_ts))
            CC = np.zeros((M, 2, 2))
            for j in range(M):
                if (z_ts == j).sum() > 0:
                    CC[j, :, :] = metrics.confusion_matrix(y_ts[z_ts == j],
                                                       ypred[z_ts == j],
                                                       labels=list(range(0,2))
                                                       ).reshape(1, 2, 2) * 1.0 / (z_ts == j).sum()
            return C, CC

    def evaluate_loss(self, loss_name, x_ts, y_ts, z_ts=None, use_stored_prob=False):
        """
        Calculate loss function

        Args:
            loss_name (string): Name of the loss function
            x_ts (array-like, dtype = float, shape = (m,d)): Test features
            y_ts (array-like, dtype = int, shape = (m,)): Test labels {0,...,m-1}
            z_ts (array-like, dtype = int, shape = (m,)): Test protected attribute {0,..M} (default: None)
            use_stored_prob (bool): Use probabilities computed from previous calls

        Returns:
            loss (float): Loss function value
        """
        module = globals()['utils']
        loss_fun = getattr(module, loss_name)
        if not self.protected_present:
            C = self.evaluate_conf(x_ts, y_ts, z_ts, use_stored_prob)
        else:
            C, _ = self.evaluate_conf(x_ts, y_ts, z_ts, use_stored_prob)
        return loss_fun(C)

    def evaluate_cons(self, cons_name, x_ts, y_ts, z_ts=None, use_stored_prob=False):
        """
        Calculate constraint function

        Args:
            cons_name (string): Name of the constraint function
            x_ts (array-like, dtype = float, shape = (m,d)): Test features
            y_ts (array-like, dtype = int, shape = (m,)): Test labels {0,...,m-1}
            z_ts (array-like, dtype = int, shape = (m,)): Test protected attribute {0,..M} (default: None)
            use_stored_prob (bool): Use probabilities computed from previous calls

        Returns:
            cons (float): Constraint function value
        """
        module = globals()['utils']
        cons_fun = getattr(module, cons_name)

        if not self.protected_present:
            C = self.evaluate_conf(x_ts, y_ts, use_stored_prob)
            return cons_fun(C)
        else:
            _, CC = self.evaluate_conf(x_ts, y_ts, z_ts, use_stored_prob)
            return cons_fun(CC)


class MulticlassPluginClassifier(PluginClassifier):
    """
    Multiclass plug-in classifier without protected attribute
    Derived from base class PluginClassifier

    Attributes:
        cpe_model (sklearn estimator): A model with a predict_proba() function
        W (array-like, dtype = float, shape = (n,n)): Cost or gain matrix
        num_class (int): Number of classes (default = 2)
        is_cost (bool): Is W a cost or gain matrix? (default = False)
        yprob (array-like, dtype = float, shape = (m,)): Stored estimated probabilities for m instances
    """
    def __init__(self, cpe_model=None, W=None, num_class=2, is_cost=True):
        """
        Initialize class

        Attributes:
            cpe_model (sklearn estimator): A model with a predict_proba() function
            W (array-like, dtype = float, shape = (n,n)): Cost or gain matrix
            num_class (int): Number of classes (default = 2)
            is_cost (bool): Is W a cost or gain matrix? (default = False)
        """
        self.cpe_model = cpe_model
        self.num_class = num_class
        self.is_cost = is_cost
        if W is None:
            if self.is_cost:
                self.W = 1 - np.identity(self.num_class)
            else:
                self.W = np.identity(self.num_class)
        else:
            self.W = W
        self.yprob = None

    def set_cost_matrix(self, W=None, is_cost=False):
        """
        Set cost (or gain) matrix

        Args:
            W (array-like, dtype = float, shape = (n,n)): Cost or gain matrix
            is_cost (bool): Is W a cost or gain matrix? (default = False)
        """
        if is_cost is not None:
            self.is_cost = is_cost
        if W is None:
            if self.is_cost:
                self.W = 1 - np.identity(self.num_class)
            else:
                self.W = np.identity(self.num_class)
        else:
            self.W = W

    def predict(self, x_ts, use_stored_prob=False):
        """
        Predict labels using plug-in classifier

        Args:
            x_ts (array-like, dtype = float, shape = (m,d)): Test features

        Returns:
            ypred (array-like, dtype = float, shape = (m,)): Predicted labels for m data points
        """
        if (not use_stored_prob) or (self.yprob is None):
            self.yprob = self.cpe_model.predict_proba(x_ts)
        m = self.yprob.shape[0]
        ypred = np.zeros((m,))
        for i in range(m):
            if self.is_cost:
                ypred[i] = np.argmin(np.dot(self.yprob[i, :], self.W))
            else:
                ypred[i] = np.argmax(np.dot(self.yprob[i, :], self.W))
        return ypred

    def predict_proba(self, x_ts):
        """
        Predict probabilities using cpe_model, and store them in yprob

        Args:
            x_ts (array-like, dtype = float, shape = (m,d)): Test features

        Returns:
            yprob (array-like, dtype = float, shape = (m,)): Predicted probabilities for m data points
        """
        self.yprob = self.cpe_model.predict_proba(x_ts)
        return self.yprob

    def evaluate_conf(self, x_ts, y_ts, use_stored_prob=False):
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
        ypred = self.predict(x_ts, use_stored_prob)
        C = metrics.confusion_matrix(y_ts, ypred, labels=list(range(self.num_class))) * 1.0 / x_ts.shape[0]
        return C

    def evaluate_loss(self, loss_name, x_ts, y_ts, use_stored_prob=False):
        """
        Calculate loss function

        Args:
            loss_name (string): Name of the loss function
            x_ts (array-like, dtype = float, shape = (m,d)): Test features
            y_ts (array-like, dtype = int, shape = (m,)): Test labels {0,...,m-1}
            use_stored_prob (bool): Use probabilities computed from previous calls

        Returns:
            loss (float): Loss function value
        """
        module = globals()['utils']
        loss_fun = getattr(module, loss_name)
        C = self.evaluate_conf(x_ts, y_ts, use_stored_prob)
        return loss_fun(C)

    def evaluate_cons(self, cons_name, x_ts, y_ts, use_stored_prob=False):
        """
        Calculate constraint function

        Args:
            cons_name (string): Name of the constraint function
            x_ts (array-like, dtype = float, shape = (m,d)): Test features
            y_ts (array-like, dtype = int, shape = (m,)): Test labels {0,...,m-1}
            use_stored_prob (bool): Use probabilities computed from previous calls

        Returns:
            cons (float): Constraint function value
        """
        module = globals()['utils']
        cons_fun = getattr(module, cons_name)
        C = self.evaluate_conf(x_ts, y_ts, use_stored_prob)
        return cons_fun(C)
