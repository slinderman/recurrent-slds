#!/usr/bin/env python

from distutils.core import setup

setup(name='rslds',
      version='0.1',
      description='Bayesian Inference for Recurrent Switching Linear Dynamical Systems',
      author='Scott W Linderman and Matthew J Johnson',
      author_email='scott.linderman@columbia.edu',
      url='http://www.github.com/slinderman/recurrent-slds',
      packages=['rslds'],
      install_requires=[
          'numpy>=1.9.3',
          'scipy>=0.16',
          'matplotlib',
          'seaborn',
          'pybasicbayes',
          'pyhsmm',
          'pylds',
          'pyslds',
          'pypolyagamma>=1.1'],
      )
