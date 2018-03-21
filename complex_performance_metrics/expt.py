import numpy as np
from time import time
import matplotlib.pyplot as plt
import gc

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from models.constrained import COCOClassifier, FRACOClassifier
from models.unconstrained import FrankWolfeClassifier, BisectionClassifier
from models.plugin import BinaryPluginClassifier, MulticlassPluginClassifier
import utils


ALGO_COCO = 0
ALGO_FRACO = 1


def batch_expts(setting_list, expt_param, solver_param):
    for (loss_name, cons_name, data_name, eps) in setting_list:
        if expt_param['verbosity']:
            print (loss_name, cons_name, data_name, eps)

        if (loss_name == 'fmeasure') or (loss_name == 'microF1'):
            solver_param['algo'] = ALGO_FRACO
        else:
            solver_param['algo'] = ALGO_COCO
        solver_param['eps_list'] = [eps]

        data = np.loadtxt('data/' + data_name + '.data', delimiter=',')

        avg_loss_cc, avg_cons_cc, avg_runtime_cc, avg_loss_cpe, avg_cons_cpe = \
            run_expt(data, loss_name, cons_name, expt_param, solver_param)

        avg_loss_unc, avg_cons_unc, avg_runtime_unc = \
            run_expt_unc(data, loss_name, cons_name, expt_param, solver_param)

        summary = '{} & {:.2f} & {:.2f} ({:.3f}) & {:.2f} ({:.3f}) & {:.2f} ({:.3f})\n'.format(
            data_name,
            eps, 1-avg_loss_cc[0], avg_cons_cc[0],
            1-avg_loss_unc, avg_cons_unc,
            1-avg_loss_cpe, avg_cons_cpe
            )

        if expt_param['verbosity']:
            print summary

        np.save('results/' + loss_name + '-' + cons_name + '-' + data_name + '-batch',
                (expt_param, solver_param,
                 avg_loss_cc, avg_cons_cc, avg_runtime_cc,
                 avg_loss_unc, avg_cons_unc, avg_runtime_unc,
                 avg_loss_cpe, avg_cons_cpe))

        result_file = open('results/batch/' + loss_name + '-' + cons_name + '-batch.txt', 'a')
        result_file.write(summary)


def plot_expts(loss_name, cons_name, data_name, expt_param, solver_param):
    if (loss_name == 'fmeasure') or (loss_name == 'microF1'):
        solver_param['algo'] = ALGO_FRACO
    else:
        solver_param['algo'] = ALGO_COCO

    data = np.loadtxt('data/' + data_name + '.data', delimiter=',')

    if expt_param['verbosity']:
        print loss_name, 's.t.', cons_name, '<= eps :', data_name
        print 'Unconstrained'

    loss_unc, cons_unc, _ = \
        run_expt_unc(data, loss_name, cons_name, expt_param, solver_param)

    max_beta = cons_unc
    eps_list = np.arange(max_beta * 1.0 / 6.0, max_beta - 0.00001, max_beta * 1.0 / expt_param['num_ticks'])

    if expt_param['verbosity']:
        print ''
        print 'eps:', eps_list

    solver_param['eps_list'] = eps_list
    loss_cc, cons_cc, _, loss_cpe, cons_cpe = \
        run_expt(data, loss_name, cons_name, expt_param, solver_param)

    # Plot figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    loss_list = {'hmean': 'H-mean Loss', 'fmeasure': 'F1 Loss', 'qmean': 'Q-mean Loss', 'microF1': 'Micro F1 Loss'}
    cons_list = {'cov': 'Coverage', 'kld': 'KLD error', 'dp': 'Demographic Parity', 'nae': 'NAE'}

    if loss_name == 'fmeasure' or loss_name == 'microF1':
        label1 = 'FRACO'
        label2 = 'BS-unc'
        label3 = 'LogReg'
    else:
        label1 = 'COCO'
        label2 = 'FW-unc'
        label3 = 'LogReg'

    ax1.plot(cons_cc, [1-x for x in loss_cc], '^', markersize=8, label=label1)
    ax1.set_xlabel(cons_list[cons_name], fontsize=14)
    ax1.set_ylabel(loss_list[loss_name], fontsize=14)
    ax1.plot(cons_unc, 1-loss_unc, 'ro', markersize=8, label=label2)  # unconstrained
    ax1.plot(cons_cpe, 1-loss_cpe, 'gs', markersize=8, label=label3)

    x_min = min(min(min(cons_cc), cons_unc), cons_cpe)
    x_max = max(max(max(cons_cc), cons_unc), cons_cpe)

    ax1.set_xlim([x_min - 0.01, x_max + 0.01])
    ax1.set_xticks(np.arange(0, x_max + 0.001, 0.2))

    ax1.legend(loc='best')

    ax2.plot(eps_list, cons_cc, 's', markersize=8)
    ax2.plot(eps_list, eps_list, 'r--', markersize=8)
    ax2.set_xlabel(r'$\epsilon$', fontsize=16)
    ax2.set_ylabel(cons_list[cons_name], fontsize=14)
    ax2.set_xlim([np.min(eps_list), np.max(eps_list)])

    plt.tight_layout()
    ax1.set_title(data_name, fontsize=14)

    plt.savefig('results/plots/' + loss_name + '-' + cons_name +
                '-' + data_name + '.eps', format='eps')

    np.save('results/plots/' + loss_name + '-' + cons_name + '-' + data_name + '-fig',
            (expt_param, solver_param, loss_cc, cons_cc, loss_unc, loss_cpe, cons_cpe))


def run_expt(data, loss_name, cons_name, expt_param, solver_param):
    # Run experiment for 5 different train-test splits
    np.random.seed(1)

    training_frac = expt_param['training_frac']
    num_trials = expt_param['num_trials']
    is_protected = expt_param['is_protected']
    verbosity = expt_param['verbosity']

    eps_list = solver_param['eps_list']
    eta_list = solver_param['eta_list']
    algo = solver_param['algo']
    num_outer_iter = solver_param['num_outer_iter']
    max_inner_ter = solver_param['num_inner_iter']

    num_class = len(np.unique(data[:, 0]))

    n = data.shape[0]
    n_tr = int(np.round(n * training_frac))

    num_eps = len(eps_list)

    avg_loss_cc = [0] * num_eps
    avg_cons_cc = [0] * num_eps
    run_times = [0] * num_eps
    avg_loss_cpe = 0
    avg_cons_cpe = 0

    gc.enable()

    for ii in range(num_trials):
        perm = np.random.permutation(n)
        if verbosity:
            print 'Trial', ii+1

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

        scaler = MinMaxScaler(copy=False)
        scaler.fit_transform(x)
        scaler.transform(x_ts)

        gc.collect()

        # Train CPE model
        cpe_model = LogisticRegressionCV(solver='liblinear')
        cpe_model.fit(x, y)

        if num_class == 2:
            if is_protected:
                M = len(np.unique(z))
                baseline_classifier = BinaryPluginClassifier(cpe_model, [0.5] * M, protected_present=True)
            else:
                baseline_classifier = BinaryPluginClassifier(cpe_model, 0.5)
        else:
            baseline_classifier = MulticlassPluginClassifier(cpe_model, num_class=num_class)

        logreg_perf = baseline_classifier.evaluate_loss(loss_name, x_ts, y_ts, z_ts)
        logreg_cons = baseline_classifier.evaluate_cons(cons_name, x_ts, y_ts, z_ts)

        avg_loss_cpe += logreg_perf * 1.0 / num_trials
        avg_cons_cpe += logreg_cons * 1.0 / num_trials

        print 'LogReg'
        print 1-logreg_perf, logreg_cons
        print

        for jj in range(num_eps):
            if eps_list[jj] == 1:
                num_eta = 1
            else:
                num_eta = len(eta_list)

            if verbosity:
                print ''
                print 'eps =', eps_list[jj]

            eta_classifier = [None] * num_eta
            eta_perf = [0] * num_eta
            eta_cons = [0] * num_eta
            eta_run_time = [0] * num_eta

            for kk in range(num_eta):
                if algo == ALGO_COCO:
                    eta_classifier[kk] = COCOClassifier(loss_name, cons_name, is_protected, num_class)
                elif algo == ALGO_FRACO:
                    eta_classifier[kk] = FRACOClassifier(loss_name, cons_name, is_protected, num_class)

                start_time = time()
                eta_classifier[kk].fit(x, y, eps_list[jj], eta_list[kk],
                                       num_outer_iter, max_inner_ter, cpe_model, z)
                eta_run_time[kk] = time() - start_time

                eta_perf[kk] = eta_classifier[kk].evaluate_loss(x, y, z)
                eta_cons[kk] = eta_classifier[kk].evaluate_cons(x, y, z)

                if verbosity:
                    print 'eta =', eta_list[kk], ':', 1-eta_perf[kk], \
                        eta_cons[kk], '(', eta_run_time[kk], 's)'
                    # C,_ = eta_classifier[kk].evaluate_conf(x, y, z)
                    # print C[0,1] + C[1,0]

            best_eta_ind = choose_best_eta(eta_perf, eta_cons, eps_list[jj], eta_list)
            best_classifier = eta_classifier[best_eta_ind]

            best_perf = best_classifier.evaluate_loss(x_ts, y_ts, z_ts)
            best_cons = best_classifier.evaluate_cons(x_ts, y_ts, z_ts)
            if verbosity:
                print 'best eta =', eta_list[best_eta_ind], ':', 1-best_perf, best_cons, '\n'

            avg_loss_cc[jj] += best_perf * 1.0 / num_trials
            avg_cons_cc[jj] += best_cons * 1.0 / num_trials
            run_times[jj] += eta_run_time[best_eta_ind] * 1.0 / num_trials

        del x, y, z, x_ts, y_ts, z_ts

    if verbosity:
        for jj in range(num_eps):
            print 'eps =', eps_list[jj], ':', 1-avg_loss_cc[jj], avg_cons_cc[jj], '(', run_times[jj], 's)'

    return avg_loss_cc, avg_cons_cc, run_times, avg_loss_cpe, avg_cons_cpe


def run_expt_unc(data, loss_name, cons_name, expt_param, solver_param):
    # Run experiment for 5 different train-test splits
    np.random.seed(1)

    training_frac = expt_param['training_frac']
    num_trials = expt_param['num_trials']
    is_protected = expt_param['is_protected']
    verbosity = expt_param['verbosity']

    algo = solver_param['algo']
    max_inner_ter = solver_param['num_inner_iter']

    num_class = len(np.unique(data[:, 0]))

    n = data.shape[0]
    n_tr = int(np.round(n * training_frac))

    avg_loss_unc = 0.0
    avg_cons_unc = 0.0
    avg_run_time = 0.0

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

        scaler = MinMaxScaler(copy=False)
        scaler.fit_transform(x)
        scaler.transform(x_ts)

        # Train CPE model
        cpe_model = LogisticRegressionCV(solver='liblinear')
        cpe_model.fit(x, y)

        if algo == ALGO_COCO:
            classifier = FrankWolfeClassifier(loss_name, protected_present=is_protected, num_class=num_class)
        else:
            classifier = BisectionClassifier(loss_name, protected_present=is_protected, num_class=num_class)

        start_time = time()
        classifier.fit(x, y, 1, 0, 1, max_inner_ter, cpe_model, z)
        run_time = time() - start_time

        perf = classifier.evaluate_loss(x_ts, y_ts, z_ts)
        if is_protected:
            _, C = classifier.evaluate_conf(x_ts, y_ts, z_ts)
        else:
            C = classifier.evaluate_conf(x_ts, y_ts, z_ts)
        cons = utils.evaluate_metric(cons_name, C)

        if verbosity:
            print 'Trial', ii+1, ':', perf, cons

        avg_loss_unc += perf * 1.0 / num_trials
        avg_cons_unc += cons * 1.0 / num_trials
        avg_run_time += run_time * 1.0 / num_trials

    if verbosity:
        print 'avg:', 1-avg_loss_unc, avg_cons_unc, '(', avg_run_time, 's)'

    return avg_loss_unc, avg_cons_unc, avg_run_time


def choose_best_eta(loss_list, cons_list, eps, eta_list):
    # Heuristic to choose index of the best eta param given their corresponding performance, constraint values
    valid_perf = [loss_list[ind] for ind in range(len(loss_list))\
                 if cons_list[ind] <= 1.05*eps]
    if len(valid_perf) > 0:
        return loss_list.index(max(valid_perf))
    return np.argmin([abs(x - eps) for x in cons_list])
