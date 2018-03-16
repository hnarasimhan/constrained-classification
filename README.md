# Learning Algorithms for Classification with Complex Losses and Constraints 

This repository contains code for the paper:

Narasimhan., H. "Learning with Complex Loss Functions and Constraints", In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics (AISTATS), 2018


## Description
We provide a package that implements learning algorithms  for two families of constrained classification problems, under complex loss functions and constraints:
- **COCO**: **Co**nvex Loss under **Co**nvex Constraints
- **FRACO**:  **Fra**ctional Convex Loss under **Co**nvex Constraints

The package contains code for optimizing the following loss functions without constraints:
- H-mean loss
- Q-mean loss
- F1-measure loss
- micro F1 loss

as well as, for solving the following constrained learning problems:
- H-mean loss subject to a *Coverage* constraint
- Q-mean loss subject to a *Demographic Parity* constraint
- F1-measure loss subject to a *KL-divergence* constraint
- Q-mean loss subject to a *Normalized Absolute Error* constraint
- micro F1 loss subject to a *Coverage* constraint


## Prerequisites
- Python (version 2.7 or higher)
- NumPy
- scikit-learn


## Usage
```
python setup.py install
```


## Created by

- [Harikrishna Narasimhan](https://sites.google.com/a/g.harvard.edu/harikrishna-narasimhan/home)


## License

MIT © [Harikrishna Narasimhan](https://sites.google.com/a/g.harvard.edu/harikrishna-narasimhan/home)
