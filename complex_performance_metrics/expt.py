import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from models.constrained import COCOClassifier, FRACOClassifier
from models.unconstrained import FrankWolfeClassifier, BisectionClassifier


"""
This module contains functions for running experiments with constrained and unconstrained classification,
    over multiple train-test splits
"""


# Define constants for COCO and FRACO
ALGO_COCO = 0
ALGO_FRACO = 1


def run_expt(loss_name, cons_name, data_name, expt_param, solver_param, eps=1):
    """
    Runs an experiment optimizing given loss function s.t. constraint
      over multiple random train-test splits,
      and returns the average loss, constraint function, and runtime values across trials

    This function supports optimizing the following loss functions without constraints:
      0-1 classification error (loss_name = 'err', cons_name = None)
      H-mean (loss_name = 'hmean', cons_name = None)
      Q-mean (loss_name = 'qmean', cons_name = None)
      F-measure (loss_name = 'fmeasure', cons_name = None)
      Micro F1 (loss_name = 'microF1', cons_name = None)

    This function supports the following binary constrained learning problems:
      0-1 classification error s.t. Demographic Parity constraint (loss_name = 'err', cons_name = 'dp')
      H-mean s.t. Coverage constraint (loss_name = 'hmean', cons_name = 'cov')
      Q-mean s.t. Demographic Parity constraint (loss_name = 'qmean', cons_name = 'dp')
      F-measure s.t. KLD constraint (loss_name = 'fmeasure', cons_name = 'kld')
    and the following multiclass constrained learning problems:
      Q-mean s.t. NAE constraint (loss_name = 'qmean', cons_name = 'nae')
      micro F1 s.t. Coverage constraint (loss_name = 'microF1', cons_name = 'cov')

    Args:
         loss_name (string): Name of loss function
                                ('er', 'hmean', 'qmean', 'fmeasure', 'microF1')
         cons_name (string): Name of constraint function
                                ('cov', 'dp', 'kld', 'nae' or None if unconstrained)
         data_name (string): Name of data set
         expt_param (dict): Dictionary of parameters for the experiment:
                            'training_frac' (float): Fraction of data set for training
                            'num_trials' (int): Number of trials with random train-test splits
                            'verbosity' (bool): Should the output be printed?
         solver_param (dict): Dictionary of parameters for the experiment:
                            'eta_list' (list): List of step-sizes eta to consider
                            'num_outer_iter': Number of outer gradient ascent iterations in COCO
                            'num_inner_iter': Number of inner Frank-Wolfe iterations in COCO
          eps (float): Constraint limit (g(h) <= eps)   (default = 1)

    Returns:
        avg_loss (float): Average loss value of learned classifier across different trials
        avg_cons (float): Average constraint value of learned classifier across different trials
        avg_runtime (float): Average runtime of the algorithm across different trials
    """
    if (loss_name == 'fmeasure') or (loss_name == 'microF1'):
        solver_param['algo'] = ALGO_FRACO
        if expt_param['verbosity']:
            print('Running FRACO for optimizing ' + loss_name +
                  (' s.t. ' + cons_name + ' constraint' if cons_name != '' else '')
                  + ' on ' + data_name + ' dataset\n')
    else:
        solver_param['algo'] = ALGO_COCO
        if expt_param['verbosity']:
            print('Running COCO for optimizing ' + loss_name +
                  (' s.t. ' + cons_name + ' constraint ' if cons_name != '' else '')
                  + ' on ' + data_name + ' dataset\n')

    # Load data set
    data = np.loadtxt('data/' + data_name + '.data', delimiter=',')

    # Run either constrained or unconstrained solver
    if cons_name != '':
        return run_expt_con(data, loss_name, cons_name, eps, expt_param, solver_param)
    else:
        return run_expt_unc(data, loss_name, expt_param, solver_param)


def run_expt_con(data, loss_name, cons_name, eps, expt_param, solver_param):
    """
    Runs experiment for constrained learning

    Args:
         data (array-like, shape=(m,d+1)):
            Data set with first column containing labels, followed by features
            (in case of a protected attribute, it must be placed as the first feature)
         loss_name (string): Name of loss function
                                ('er', 'hmean', 'qmean', 'fmeasure', 'microF1')
         cons_name (string): Name of constraint function
                                ('cov', 'dp', 'kld', 'nae' or None if unconstrained)
         eps (float): Constraint limit (g(h) <= eps)
         expt_param (dict): Dictionary of parameters for the experiment (see docstring for run_expt())
         solver_param (dict): Dictionary of parameters for the experiment (see docstring for run_expt())

    Returns:
        avg_loss (float): Average loss value of learned classifier across different trials
        avg_cons (float): Average constraint value of learned classifier across different trials
        avg_runtime (float): Average runtime of the algorithm across different trials
    """
    np.random.seed(1) # Set random seed

    training_frac = expt_param['training_frac']
    num_trials = expt_param['num_trials']
    is_protected = expt_param['is_protected']
    verbosity = expt_param['verbosity']

    eta_list = solver_param['eta_list']
    algo = solver_param['algo']
    num_outer_iter = solver_param['num_outer_iter']
    max_inner_ter = solver_param['num_inner_iter']

    num_class = len(np.unique(data[:, 0]))
    num_eta = len(eta_list)

    # Calculate number of training points
    n = data.shape[0]
    n_tr = int(np.round(n * training_frac))

    if verbosity:
        print('\n' + 'eps = ' + str(eps) + '\n')

    avg_loss = 0
    avg_cons = 0
    avg_runtime = 0

    # Run for specified number of trials
    for ii in range(num_trials):
        # Permute data, and split into train and test sets
        perm = np.random.permutation(n)
        if verbosity:
            print('Trial ' + str(ii+1))

        y = data[perm[0:n_tr], 0]
        x = data[perm[0:n_tr], 1:]
        if is_protected:
            z = data[perm[0:n_tr], 1]
        else:
            z = None

        y_ts = data[perm[n_tr:], 0]
        x_ts = data[perm[n_tr:], 1:]
        if is_protected:
            z_ts = data[perm[n_tr:], 1]
        else:
            z_ts = None

        # Scale train set and apply same transformation to test set
        scaler = MinMaxScaler(copy=False)
        scaler.fit_transform(x)
        scaler.transform(x_ts)

        # Train base class probability estimation model
        cpe_model = LogisticRegressionCV(solver='liblinear')
        cpe_model.fit(x, y)

        eta_classifier = [None] * num_eta
        eta_loss = [0] * num_eta
        eta_cons = [0] * num_eta
        eta_run_time = [0] * num_eta

        # Try different values of eta, and record the loss, constraint values and the classifier
        for kk in range(num_eta):
            # Choose between COCO and FRACO models
            if algo == ALGO_COCO:
                eta_classifier[kk] = COCOClassifier(loss_name, cons_name, is_protected, num_class)
            elif algo == ALGO_FRACO:
                eta_classifier[kk] = FRACOClassifier(loss_name, cons_name, is_protected, num_class)

            # Fit classifier using given eta value, keep track of the run time
            start_time = time()
            eta_classifier[kk].fit(x, y, eps, eta_list[kk],
                                   num_outer_iter, max_inner_ter, cpe_model, z)
            eta_run_time[kk] = time() - start_time

            eta_loss[kk] = eta_classifier[kk].evaluate_loss(x, y, z)
            eta_cons[kk] = eta_classifier[kk].evaluate_cons(x, y, z)

            if verbosity:
                print('eta = ' + str(eta_list[kk]) + ' : ' + str(eta_loss[kk]) + ' / ' +
                      str(eta_cons[kk]) + ' (' + str(eta_run_time[kk]) + 's)')

        # Choose the best eta based stats recorded
        best_eta_ind = choose_best_eta(eta_loss, eta_cons, eps)
        best_classifier = eta_classifier[best_eta_ind]

        # Evaluate the classifier at the best eta
        best_loss = best_classifier.evaluate_loss(x_ts, y_ts, z_ts)
        best_cons = best_classifier.evaluate_cons(x_ts, y_ts, z_ts)
        if verbosity:
            print('best eta = ' + str(eta_list[best_eta_ind]) + ' : ' + str(best_loss)
                  + ' / ' + str(best_cons) + '\n')

        avg_loss += best_loss * 1.0 / num_trials
        avg_cons += best_cons * 1.0 / num_trials
        avg_runtime += eta_run_time[best_eta_ind] * 1.0 / num_trials

    if verbosity:
        print('eps = ' + str(eps) + ' : ' + str(avg_loss) + ' / ' + str(avg_cons) +
              ' (' + str(avg_runtime) + 's)')

    return avg_loss, avg_cons, avg_runtime


def run_expt_unc(data, loss_name, expt_param, solver_param):
    """
    Runs experiment for unconstrained learning

    Args:
         data (array-like, shape=(m,d+1)):
            Data set with first column containing labels, followed by features
            (in case of a protected attribute, it must be placed as the first feature)
         loss_name (string): Name of loss function
                                ('er', 'hmean', 'qmean', 'fmeasure', 'microF1')
         expt_param (dict): Dictionary of parameters for the experiment (see docstring for run_expt())
         solver_param (dict): Dictionary of parameters for the experiment:
                            'num_inner_iter' (int):
                                Number of iterations in Frank-Wolfe, specify None for Bisection algorithm

    Returns:
        avg_loss (float): Average loss value of learned classifier across different trials
        avg_runtime (float): Average runtime of the algorithm across different trials
    """
    np.random.seed(1) # Set random seed

    training_frac = expt_param['training_frac']
    num_trials = expt_param['num_trials']
    is_protected = expt_param['is_protected']
    verbosity = expt_param['verbosity']

    algo = solver_param['algo']
    num_iter = solver_param['num_inner_iter']

    num_class = len(np.unique(data[:, 0]))

    # Calculate number of training points
    n = data.shape[0]
    n_tr = int(np.round(n * training_frac))

    avg_loss = 0.0
    avg_runtime = 0.0

    # Run for specified number of trials
    for ii in range(num_trials):
        perm = np.random.permutation(n)

        y = data[perm[0:n_tr], 0]
        x = data[perm[0:n_tr], 1:]
        if is_protected:
            z = data[perm[0:n_tr], 1]
        else:
            z = None

        y_ts = data[perm[n_tr:], 0]
        x_ts = data[perm[n_tr:], 1:]
        if is_protected:
            z_ts = data[perm[n_tr:], 1]
        else:
            z_ts = None

        # Scale train set and apply same transformation to test set
        scaler = MinMaxScaler(copy=False)
        scaler.fit_transform(x)
        scaler.transform(x_ts)

        # Train a base class-probability estimation model
        cpe_model = LogisticRegressionCV(solver='liblinear')
        cpe_model.fit(x, y)

        # Fit classifier using either Frank-Wolfe or Bisection algorithm, keep track of run-time
        if algo == ALGO_COCO:
            classifier = FrankWolfeClassifier(loss_name, protected_present=is_protected, num_class=num_class)
        else:
            classifier = BisectionClassifier(loss_name, protected_present=is_protected, num_class=num_class)

        start_time = time()
        classifier.fit(x, y, num_iter, cpe_model, z)
        run_time = time() - start_time

        #
        loss = classifier.evaluate_loss(x_ts, y_ts, z_ts)
        if verbosity:
            print('Trial' + str(ii+1) + ' : ' + str(loss))

        avg_loss += loss * 1.0 / num_trials
        avg_runtime += run_time * 1.0 / num_trials

    if verbosity:
        print('unconstrained: ' + str(avg_loss) + ' (' + str(avg_runtime) + 's)')

    return avg_loss, avg_runtime


def choose_best_eta(loss_list, cons_list, eps):
    """
    Heuristic to choose index of the best eta param given their loss, constraint values and eps:
        If there is an eta for which constraint value <= 1.05 * eps
            Among all such eta, choose the one with minimum loss
        Else:
            Choose eta that minimizes |constraint value - eps|
    Args:
        loss_list (list, dtype = float): List of loss function values for different eta values
        cons_list (list, dtype = float): List of corresponding constraint function values
        eps (float): Constraint limit (g(h) <= eps)

    Returns:
        eta_ind (int): Index of chosen value of eta
    """
    valid_loss = [loss_list[ind] for ind in range(len(loss_list))\
                 if cons_list[ind] <= 1.05*eps]
    if len(valid_loss) > 0:
        return loss_list.index(min(valid_loss))
    return np.argmin([abs(x - eps) for x in cons_list])
