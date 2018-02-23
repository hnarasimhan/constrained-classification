from randomized_classifier import RandomizedClassifier
import complex_performance_metrics.solvers as solvers


class UnconstrainedClassifier(RandomizedClassifier):
    # Base class
    def __init__(self, perf_name, protected_present=False, num_class=2):
        RandomizedClassifier.__init__(self, perf_name, solvers.PERF_CONS_MAP[perf_name], protected_present, num_class)

    def fit(self, x, y, eps, eta, max_outer_iter, max_inner_iter, cpe_model=None, z=None):
        RandomizedClassifier.fit(self, x, y, eps, 1, max_outer_iter, max_inner_iter, cpe_model, z)


class FrankWolfeClassifier(UnconstrainedClassifier):
    # COCO for optimizing convex losses
    def __init__(self, perf_name, protected_present=False, num_class=2):
        UnconstrainedClassifier.__init__(self, perf_name, protected_present, num_class)
        self.opt_name = 'coco'


class BisectionClassifier(UnconstrainedClassifier):
    # FRACO for optimizing fractional-convex losses
    def __init__(self, perf_name, protected_present=False, num_class=2):
        UnconstrainedClassifier.__init__(self, perf_name, protected_present, num_class)
        self.opt_name = 'fraco'


class LinClassifier(UnconstrainedClassifier):
    # COCO for optimizing linear losses
    def __init__(self, perf_name, protected_present=False, num_class=2):
        UnconstrainedClassifier.__init__(self, perf_name, protected_present, num_class)
        self.opt_name = 'err'
