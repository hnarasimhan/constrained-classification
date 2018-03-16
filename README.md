## Learning Algorithms for Classification with Complex Losses and Constraints 

This repository contains code for the paper:

Narasimhan., H. "Learning with Complex Loss Functions and Constraints", In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS), 2018

The repository also implements algorithms from a previous paper:
Narasimhan, H.*, Ramaswamy, H. G.*, Saha, A. and Agarwal, S. 'Consistent multiclass algorithms for complex performance measures'. In Proceedings of the 32nd International Conference on Machine Learning (ICML), 2015. 
(*both authors contributed equally to the paper)


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
```

```


## Created by

- [Harikrishna Narasimhan](https://sites.google.com/a/g.harvard.edu/harikrishna-narasimhan/home)


## License

MIT © [Harikrishna Narasimhan](https://sites.google.com/a/g.harvard.edu/harikrishna-narasimhan/home)
