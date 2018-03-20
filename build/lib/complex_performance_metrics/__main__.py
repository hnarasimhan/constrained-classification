import warnings
from expt import batch_expts, plot_expts

expt_param = {'training_frac': 2.0/3.0, 'num_trials': 5, 'verbosity': True, 'num_ticks': 6}

solver_param = {'eta_list': [0.01,0.1,1,10,100,1000], 'max_inner_iter': 10, 'max_outer_iter': 100}

warnings.filterwarnings("ignore")

data = raw_input("dataset: ")
perf = raw_input("performance measure (hmean/qmean/fmeasure/microF1): ")
cons = raw_input("constraint function (cov/dp/kld): ")
protected = raw_input("is there a protected attribute (true/false): ")

expt_param['is_protected'] = True if protected == 'true' else False

out_type = raw_input("output type (text/plot): ")
if out_type == 'text':
    eps = raw_input("eps: ")
    batch_expts([(perf, cons, data, float(eps))], expt_param, solver_param)
else:
    plot_expts(perf, cons, data, expt_param, solver_param)