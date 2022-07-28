"""
Copyright (c) 2017, Stanford University.
All rights reserved.

This source code is a Python implementation of SIMLR for the following paper published in Nature Methods:
Visualization and analysis of single-cell RNA-seq data by kernel-based similarity learning
"""
from distutils.core import setup

setup(
    name='simlr_py3',
    version='0.1.5',
    author='Xuhang Chen',
    author_email='cxhhg11@gmail.com',
    url='https://github.com/SIAT-BIT-CXH/SIMLR_PY',
    description='Visualization and analysis of single-cell RNA-seq data by kernel-based similarity learning',
    packages=['simlr_py3'],
    install_requires=[
                   'fbpca>=1.0',
                   'numpy>=1.8.0',
                   'scipy>=0.13.2',
                   'annoy>=1.8.0',
                   'scikit-learn>=0.17.1',
     ],
    classifiers=[])
