from setuptools import setup

setup(
    name='complex_performance_metrics',
    version='1.0',
    packages=['complex_performance_metrics', 'complex_performance_metrics.models',
              'complex_performance_metrics.solvers', 'complex_performance_metrics.solvers.binary',
              'complex_performance_metrics.solvers.multiclass'],
    url='',
    license='MIT',
    author='Harikrishna Narasimhan',
    author_email='hnarasimhan@seas.harvard.edu',
    description='Algorithms for learning with complex loss functions and constraints'
)
