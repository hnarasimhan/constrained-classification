import numpy as np


# Generic function call for performance measure / constraints
def evaluate_metric(metric_name, C):
        metric_fun = globals()[metric_name]
        return metric_fun(C)


# Function definitions for performance measures
def err(C):
    # Calculate classification error
    return 1 - np.trace(C)


def hmean(C):
    # Calculate H-mean
    tpr = C[1, 1] * 1.0 / (C[1, 0] + C[1, 1])
    tnr = C[0, 0] * 1.0 / (C[0, 0] + C[0, 1])
    return 2.0 * tpr * tnr / (tpr + tnr)


def fmeasure(C):
   # Calculate F1-measure
   if C[0, 1] + C[1, 1] > 0:
       prec = C[1, 1] * 1.0 / (C[0, 1] + C[1, 1])
   else:
       prec = 1.0
   rec = C[1, 1] * 1.0 / (C[1, 0] + C[1, 1])
   if prec + rec == 0:
       return 1.0
   else:
       return 2 * prec * rec / (prec + rec)


def qmean(C):
    # Calculate multi-class Q-mean
    n = C.shape[0]
    qm = 0.0
    for i in range(n):
        qm = qm + (1 - C[i, i] / C[i, :].sum()) ** 2 / n
    return 1-np.sqrt(qm)


def microF1(C):
    # Calculate multi-class microF1
    n = C.shape[0]
    num = 0
    for i in range(1, n):
        num += 2.0 * C[i, i]
    dem = 2 - C[0, :].sum() - C[:, 0].sum()
    if dem == 0:
        return 1.0
    else:
        return num / (2 - C[0, :].sum() - C[:, 0].sum())


# Function definitions for constraint functions
def cov(C):
    # Calculate Coverage
    return 1 - C[:, 0].sum() # C[0, 1] + C[1, 1]


def dp(CC):
   # Calculate Demographic Parity
   M = CC.shape[0]
   C_mean = np.zeros((2, 2))
   dparity = np.zeros((M, 1))
   for j in range(M):
       C_mean += CC[j, :, :].reshape((2, 2)) * 1.0 / M
   for j in range(M):
       dparity[j] = CC[j, 0, 1] + CC[j, 1, 1] - C_mean[0, 1] - C_mean[1, 1]
   return np.abs(dparity).max()


def eo(C0, C1):
    # Calculate Equal Odds
    return np.max([np.abs(C0[0, 1] / C0[0, :].sum() - C1[0, 1] / C1[0, :].sum()),
                   np.abs(C0[1, 0] / C0[1, :].sum() - C1[1, 0] / C1[1, :].sum())])  # / (1-p) / p) * 0.5


def kld(C):
    # Calculate KL-divergence quantization loss
    eps = 0.0001
    C = C * 1.0 / C.sum()
    p = max([min([C[1, :].sum(), 1 - eps]), eps])
    phat = max([min([C[:, 1].sum(), 1 - eps]), eps])
    return p * np.log(p / phat) + (1 - p) * np.log((1 - p) / (1 - phat))


def nae(C):
    # Calculate multi-class NAE quantization measure
    n = C.shape[0]
    er = 0.0
    p = np.sum(C, axis=1)
    for i in range(n):
        er += np.abs(C[:, i].sum() - p[i])  # / n
    return er * 0.5 / (1 - np.min(p))