# Map from performance measure to constraint functions
PERF_CONS_MAP = {'hmean': 'cov', 'qmean': 'dp', 'fmeasure': 'kld', 'microF1':'cov', 'err': 'dp'}
CONS_UPPER = {'cov': 1.0, 'kld': 1.0, 'dp': 1.0, 'nae': 1.0}