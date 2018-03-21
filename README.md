## Learning Algorithms for Classification with Complex Losses and Constraints 

This repository contains code for the paper:

Narasimhan., H. "Learning with Complex Loss Functions and Constraints", In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS), 2018

The repository also implements algorithms from a previous paper:

Narasimhan, H., Ramaswamy, H. G., Saha, A. and Agarwal, S. 'Consistent multiclass algorithms for complex performance measures'. In Proceedings of the 32nd International Conference on Machine Learning (ICML), 2015.

(the first two authors are equal contributors)


## Description
We provide a package `complex_performance_metrics` that implements the following two algorithms from Narasimhan et al. (2015) for optimizing complex loss functions without constraints:
- **Frank-Wolfe** based algorithm for optimizing convex loss functions
- **Bisection** based algorithm for optimizing fractional-convex loss functions

and covers the following loss functions:
- H-mean loss
- Q-mean loss
- F1-measure loss
- micro F1 loss

The package implements the following algorithms from Narasimhan (2018) for two families of constrained classification problems, under complex loss functions and constraints:
- **COCO**: **Co**nvex Loss under **Co**nvex Constraints
- **FRACO**:  **Fra**ctional Convex Loss under **Co**nvex Constraints

and covers the following constrained learning problems:
- 0-1 classification loss subject to a *Demographic Parity* constraint
- H-mean loss subject to a *Coverage* constraint
- Q-mean loss subject to a *Demographic Parity* constraint
- F1-measure loss subject to a *KL-divergence* constraint
- Q-mean loss subject to a *Normalized Absolute Error* constraint
- micro F1 loss subject to a *Coverage* constraint


## Prerequisites
- Python (version 2.7 or higher)
- NumPy
- scikit-learn


## Install
To install the package, run the following command in the source directory:
```
python setup.py install
```

## Example Usage
The package `models.unconstrained` allows you to fit models that optimize unconstrained complex losses and the package `models.constrained` allows you to  fit models that optimize complex losses under constraints. We begin by first fitting a class probability estimation model such as a logisitc regression to a given NumPy arrays of instances `x` and labels `y`:

```
from sklearn.linear_model import LogisticRegressionCV

cpe_model = LogisticRegressionCV(solver='liblinear')
cpe_model.fit(x, y)
```

The following code snippet creates a `FrankWolfeClassifier` object to fit a model that optimizes a convex loss function and evaluate its loss:
```
from models.unconstrained import FrankWolfeClassifier

classifier = FrankWolfeClassifier('hmean')
classifier.fit(x, y, eps = 0.1, eta = 0.1, num_outer_iter=100, num_inner_iter=10, cpe_model=cpe_model)

hmean_loss = classifier.evaluate_loss(x, y)
```

The following code snippet creates a `BisectionClassifier` object to fit a model that optimizes a fractional-convex loss function, and to evaluate its loss:
```
from models.unconstrained import BisectionClassifier

classifier = BisectionClassifier('fmeasure')
classifier.fit(x, y, eps = 0.1, eta = 0.1, num_outer_iter=10, cpe_model=cpe_model)

f1_loss = classifier.evaluate_loss(x, y)
```

The following code snippet creates a `COCOClassifier` object to fit a model that optimizes a convex loss function subject to a convex constraint function, and to evaluate its loss and constraint values:
```
from models.constrained import COCOClassifier

classifier = COCOClassifier('hmean', 'cov')
classifier.fit(x, y, eps = 0.1, eta = 0.1, num_outer_iter=100, num_inner_iter=10, cpe_model=cpe_model)

hmean_loss = classifier.evaluate_loss(x, y)
cov = classifier.evaluate_cons(x, y)
```

The following code snippet creates a `FRACOClassifier` object to fit a model that optimizes a convex loss function subject to a fractional-convex constraint function, and to evaluate its loss and constraint values:
```
from models.constrained import FRACOClassifier

classifier = FRACIClassifier('fmeasure', 'kld')
classifier.fit(x, y, eps = 0.1, eta = 0.1, num_outer_iter=100, num_inner_iter=10, cpe_model=cpe_model)

f1_loss = classifier.evaluate_loss(x, y)
kld = classifier.evaluate_cons(x, y)
```

## Demo Code
The file `complex_performance_metrics/run_demo.py` contains demo code for running experiments with constrained and unconstrained classification on the different data set in the folder `complex_performance_metrics/data`. The code takes user inputs from terminal and can be executed using:
```
python run_demo.py
```


## Created by

- [Harikrishna Narasimhan](https://sites.google.com/a/g.harvard.edu/harikrishna-narasimhan/home)


## License

MIT © [Harikrishna Narasimhan](https://sites.google.com/a/g.harvard.edu/harikrishna-narasimhan/home)
