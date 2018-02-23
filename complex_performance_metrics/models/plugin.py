import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import complex_performance_metrics.utils as utils


class BinaryPluginClassifier:
    # Binary plug-in classifier
    def __init__(self, cpe_model=None, t0=0.5, t1=None, protected_present=False):
        self.cpe_model = cpe_model
        self.t0 = t0
        self.t1 = t1
        self.yprob = None
        self.protected_present = protected_present

    def fit_cpe(self, x, y):
        # Fit Logistic regression cpe model to data
        self.cpe_model = LogisticRegressionCV(solver='liblinear')
        self.cpe_model.fit(x, y)

    def set_cpe(self, cpe_model):
        self.cpe_model = cpe_model

    def set_thresh(self, t0=0.5, t1=None):
        # Set thresholds for plug-in classifier
        self.t0 = t0
        self.t1 = t1

    def predict(self, x_ts, z_ts=None, use_stored_prob=False):
        # Compute predictions from scores, protected attribute, thresholds
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
        # Predict with probabilities using cpe_model
        self.yprob = self.cpe_model.predict_proba(x_ts)[:,1]
        return self.yprob

    def evaluate_conf(self, x_ts, y_ts, z_ts=None, use_stored_prob=False):
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
            # C0 = metrics.confusion_matrix(y_ts[z_ts == 0], ypred[z_ts == 0]) * 1.0 / np.sum(z_ts==0)
            # C1 = metrics.confusion_matrix(y_ts[z_ts == 1], ypred[z_ts == 1]) * 1.0 / np.sum(z_ts==1)
            # return C, C0, C1

    def evaluate_perf(self, perf_name, x_ts, y_ts, z_ts=None, use_stored_prob=False):
        module = globals()['utils']
        perf_fun = getattr(module, perf_name)
        if not self.protected_present:
            C = self.evaluate_conf(x_ts, y_ts, z_ts, use_stored_prob)
        else:
            C, _ = self.evaluate_conf(x_ts, y_ts, z_ts, use_stored_prob)
            # C,_,_ = self.evaluate_conf(x_ts, y_ts, z_ts, use_stored_prob)
        return perf_fun(C)

    def evaluate_cons(self, cons_name, x_ts, y_ts, z_ts=None, use_stored_prob=False):
        module = globals()['utils']
        cons_fun = getattr(module, cons_name)

        if not self.protected_present:
            C = self.evaluate_conf(x_ts, y_ts, use_stored_prob)
            return cons_fun(C)
        else:
            _, CC = self.evaluate_conf(x_ts, y_ts, z_ts, use_stored_prob)
            return cons_fun(CC)
            # C, C0, C1 = self.evaluate_conf(x_ts, y_ts, z_ts, use_stored_prob)
            # return cons_fun(C0, C1)


class MulticlassPluginClassifier:
    # Muliclass plug-in classifier
    def __init__(self, cpe_model=None, W=None, num_class=2, is_cost=True):
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

    def fit_cpe(self, x, y):
        # Fit Logistic regression cpe model to data
        self.cpe_model = LogisticRegressionCV(solver='liblinear')
        self.cpe_model.fit(x, y)

    def set_cpe(self, cpe_model):
        self.cpe_model = cpe_model

    def set_cost_matrix(self, W=None, is_cost=None):
        # Set cost matrix for plug-in classifier
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
        # Compute predictions from scores, protected attribute, thresholds
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
        # Predict with probabilities using cpe_model
        self.yprob = self.cpe_model.predict_proba(x_ts)
        return self.yprob

    def evaluate_conf(self, x_ts, y_ts, use_stored_prob=False):
        ypred = self.predict(x_ts, use_stored_prob)
        C = metrics.confusion_matrix(y_ts, ypred, labels=list(range(self.num_class))) * 1.0 / x_ts.shape[0]
        return C

    def evaluate_perf(self, perf_name, x_ts, y_ts, use_stored_prob=False):
        module = globals()['utils']
        perf_fun = getattr(module, perf_name)
        C = self.evaluate_conf(x_ts, y_ts, use_stored_prob)
        return perf_fun(C)

    def evaluate_cons(self, cons_name, x_ts, y_ts, use_stored_prob=False):
        module = globals()['utils']
        cons_fun = getattr(module, cons_name)
        C = self.evaluate_conf(x_ts, y_ts, use_stored_prob)
        return cons_fun(C)
