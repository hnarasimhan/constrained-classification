import numpy as np

data = np.loadtxt('data/crimes.data', delimiter=',')

y = data[:, 0]
x = data[:, 1:]

from sklearn.linear_model import LogisticRegressionCV

cpe_model = LogisticRegressionCV(solver='liblinear')
cpe_model.fit(x, y)

from models.unconstrained import FrankWolfeClassifier

classifier = FrankWolfeClassifier('hmeana')
classifier.fit(x, y, num_outer_iter=10, cpe_model=cpe_model)

hmean_loss = classifier.evaluate_loss(x, y)

print hmean_loss

from models.unconstrained import BisectionClassifier

classifier = BisectionClassifier('microF1')
classifier.fit(x, y, cpe_model=cpe_model)

f1_loss = classifier.evaluate_loss(x, y)

print f1_loss

from models.constrained import FRACOClassifier

classifier = FRACOClassifier('microF1', 'dp')
classifier.fit(x, y, 0.1, 1, 100, 10, cpe_model=cpe_model)

f1_loss = classifier.evaluate_loss(x, y)
cov = classifier.evaluate_cons(x, y)

print f1_loss, cov