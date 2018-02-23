from randomized_classifier import RandomizedClassifier
import complex_performance_metrics.utils as utils


class ConstrainedClassifier(RandomizedClassifier):
    # Base class
    def __init__(self, perf_name, cons_name, protected_present=False, num_class=2):
        RandomizedClassifier.__init__(self, perf_name, cons_name, protected_present, num_class)

    def fit(self, x, y, eps, eta, max_outer_iter, max_inner_iter, cpe_model=None, z=None):
        RandomizedClassifier.fit(self, x, y, eps, eta, max_outer_iter, max_inner_iter, cpe_model, z)

    def evaluate_cons(self, x_ts, y_ts, z_ts=None):
        if not self.protected_present:
            C = self.evaluate_conf(x_ts, y_ts)
        else:
            _, C = self.evaluate_conf(x_ts, y_ts, z_ts)
        return utils.evaluate_metric(self.cons_name, C)


class COCOClassifier(ConstrainedClassifier):
    # COCO for optimizing convex losses s.t. convex constraints
    def __init__(self, perf_name, cons_name, protected_present=False, num_class=2):
        ConstrainedClassifier.__init__(self, perf_name, cons_name, protected_present, num_class)
        self.opt_name = 'coco'


class FRACOClassifier(ConstrainedClassifier):
    # FRACO for optimizing fractional-convex losses s.t. convex constraints
    def __init__(self, perf_name, cons_name, protected_present=False, num_class=2):
        ConstrainedClassifier.__init__(self, perf_name, cons_name, protected_present, num_class)
        self.opt_name = 'fraco'


