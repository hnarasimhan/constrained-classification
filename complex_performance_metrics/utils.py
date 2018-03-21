import numpy as np


"""
This module contains functions for computing different loss and constraint functions
"""

def evaluate_metric(metric_name, C):
    """
    Generic function to evaluate a metric

    Attributes:
        metric_name (string): Name of metric
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        metric (float): Metric value
    """
    if metric_name not in globals():
        raise KeyError('Metric name not found')
    metric_fun = globals()[metric_name]
    return metric_fun(C)


def err(C):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        loss (float): 0-1 classification error
    """
    return 1 - np.trace(C)


def hmean(C):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        loss (float): H-mean loss
    """
    tpr = C[1, 1] * 1.0 / (C[1, 0] + C[1, 1])
    tnr = C[0, 0] * 1.0 / (C[0, 0] + C[0, 1])
    return 1.0 - 2.0 * tpr * tnr / (tpr + tnr)


def fmeasure(C):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        loss (float): F-measure loss
    """
    if C[0, 1] + C[1, 1] > 0:
       prec = C[1, 1] * 1.0 / (C[0, 1] + C[1, 1])
    else:
       prec = 1.0
    rec = C[1, 1] * 1.0 / (C[1, 0] + C[1, 1])
    if prec + rec == 0:
       return 0.0
    else:
       return 1.0 - 2 * prec * rec / (prec + rec)


def qmean(C):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        loss (float): Q-mean loss
    """
    n = C.shape[0]
    qm = 0.0
    for i in range(n):
        qm = qm + (1 - C[i, i] / C[i, :].sum()) ** 2 / n
    return np.sqrt(qm)


def microF1(C):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        loss (float): microF1 loss
    """
    n = C.shape[0]
    num = 0
    for i in range(1, n):
        num += 2.0 * C[i, i]
    dem = 2 - C[0, :].sum() - C[:, 0].sum()
    if dem == 0:
        return 0.0
    else:
        return 1.0 - num / dem


def cov(C):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        cons (float): Coverage constraint function value
    """
    return 1 - C[:, 0].sum() # C[0, 1] + C[1, 1]


def dp(CC):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        cons (float): Demographic parity constraint function value
    """
    M = CC.shape[0]
    C_mean = np.zeros((2, 2))
    dparity = np.zeros((M, 1))
    for j in range(M):
       C_mean += CC[j, :, :].reshape((2, 2)) * 1.0 / M
    for j in range(M):
       dparity[j] = CC[j, 0, 1] + CC[j, 1, 1] - C_mean[0, 1] - C_mean[1, 1]
    return np.abs(dparity).max()


def kld(C):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        cons (float): Binary KL-divergence constraint value
    """
    eps = 0.0001
    C = C * 1.0 / C.sum()
    p = max([min([C[1, :].sum(), 1 - eps]), eps])
    phat = max([min([C[:, 1].sum(), 1 - eps]), eps])
    return p * np.log(p / phat) + (1 - p) * np.log((1 - p) / (1 - phat))


def nae(C):
    """
    Attributes:
        C (array-like, dtype=float, shape=(n,n)): Confusion matrix

    Returns:
        cons (float): Multi-class normalized absolute error constraint value
    """
    n = C.shape[0]
    er = 0.0
    p = np.sum(C, axis=1)
    for i in range(n):
        er += np.abs(C[:, i].sum() - p[i])
    return er * 0.5 / (1 - np.min(p))
