import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from complex_performance_metrics.solvers.binary import \
    coco_hmean_cov, coco_qmean_dp, coco_err_dp, fraco_fmeasure_kld
from complex_performance_metrics.solvers.multiclass import \
    coco_qmean_nae, fraco_microF1_cov
import complex_performance_metrics.utils as utils


class RandomizedClassifier:
    def __init__(self, perf_name, cons_name, protected_present=False, num_class=2):
        self.weights = []
        self.classifiers = list()
        self.protected_present = protected_present
        self.num_class = num_class
        self.perf_name = perf_name
        self.cons_name = cons_name
        self.opt_name = None

    def append(self, w, plugin):
        if type(w) == list:
            self.weights.append(w)
            self.classifiers.append(plugin)
        else:
            self.weights += [w]
            self.classifiers += [plugin]

    def normalize_weights(self):
        tot = sum(self.weights)
        if tot>0:
            self.weights = [x*1.0/tot for x in self.weights]

    def evaluate_conf(self, x_ts, y_ts, z_ts=None, use_stored_prob=False):
        # Is protected attribute present
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

    def fit(self, x, y, eps, eta, max_outer_iter, max_inner_iter, cpe_model=None, z=None):
        # Fit randomized classifier to data
        if cpe_model is None:
            cpe_model = LogisticRegressionCV()
            cpe_model.fit(x, y)

        # Get module for performance measure / constraint
        module = globals()[self.opt_name + '_' + self.perf_name + '_' + self.cons_name]

        # Check if there is a protected attribute
        if not self.protected_present:
            module.fit(x, y, self, cpe_model, eps, eta, max_outer_iter, max_inner_iter)
        else:
            self.protected_present = True
            module.fit(x, y, z, self, cpe_model, eps, eta, max_outer_iter, max_inner_iter)

    def evaluate_perf(self, x_ts, y_ts, z_ts=None):
        if not self.protected_present:
            C = self.evaluate_conf(x_ts, y_ts)
        else:
            C, _ = self.evaluate_conf(x_ts, y_ts, z_ts)
        return utils.evaluate_metric(self.perf_name, C)
