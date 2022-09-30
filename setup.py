#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Online algorithm configuration for differential evolution algorithm (OAC-DE)

    Copyright 2022 Changwu Huang, Hao Bai and Xin Yao

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
'''
from setuptools import setup, find_packages

setup(
    name='oacde',
    version='0.0.1',
    url='https://github.com/sustech-opal/oac-de',
    author='Changwu Huang, Hao Bai and Xin Yao',
    author_email='huangcw3@sustech.edu.cn',
    maintainer='Hao Bai',
    maintainer_email='hao.bai@insa-rouen.fr',
    license='Apache License Version 2.0, January 2004',
    platforms='py38-none-linux_x86_64',
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'License :: Free For Educational Use',
    ],

    keywords='Automatic algorithm configuration, Differential evolution algorithm, Adaptive parameter control, Multi-armed bandit, Kernel density estimation, Machine learning',
    description='Online algorithm configuration for differential evolution algorithm (OAC-DE)',
    long_description="The performance of evolutionary algorithms (EAs) is strongly affected by their configurations. Thus, algorithm configuration (AC) problem, that is, to properly set algorithm's configuration, including the operators and parameter values for maximizing the algorithm's performance on given problem(s) is an essential and challenging task in the design and application of EAs. In this paper, an online algorithm configuration (OAC) approach is proposed for differential evolution (DE) algorithm to adapt its configuration in a data-driven way. In our proposed OAC, the multi-armed bandit algorithm is adopted to select trial vector generation strategies for DE, and the kernel density estimation method is used to adapt the associated control parameters during the evolutionary search process. The performance of DE algorithm using the proposed OAC (OAC-DE) is evaluated on a benchmark set of 30 bound-constrained numerical optimization problems and compared with several adaptive DE variants. Besides, the influence of OAC's hyper-parameter on its performance is analyzed. The comparison results show OAC-DE achieves better average performance than the compared algorithms, which validates the effectiveness of the proposed OAC. The sensitivity analysis indicates that the hyper-parameter of OAC has little impact on OAC-DE's performance.",

    readme='README.md',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'matplotlib',
    ],
    zip_safe=False,
)