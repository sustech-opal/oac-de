# Online algorithm configuration for differential evolution algorithm (OAC-DE)

[Click here to read the full paper online.](https://link.springer.com/article/10.1007/s10489-021-02752-1)

[Click here to download the full paper in PDF file.](https://rdcu.be/cV80v)

## Abstract

The performance of evolutionary algorithms (EAs) is strongly affected by their configurations. Thus, algorithm configuration (AC) problem, that is, to properly set algorithm's configuration, including the operators and parameter values for maximizing the algorithm's performance on given problem(s) is an essential and challenging task in the design and application of EAs. In this paper, an online algorithm configuration (OAC) approach is proposed for differential evolution (DE) algorithm to adapt its configuration in a data-driven way. In our proposed OAC, the multi-armed bandit algorithm is adopted to select trial vector generation strategies for DE, and the kernel density estimation method is used to adapt the associated control parameters during the evolutionary search process. The performance of DE algorithm using the proposed OAC (OAC-DE) is evaluated on a benchmark set of 30 bound-constrained numerical optimization problems and compared with several adaptive DE variants. Besides, the influence of OAC's hyper-parameter on its performance is analyzed. The comparison results show OAC-DE achieves better average performance than the compared algorithms, which validates the effectiveness of the proposed OAC. The sensitivity analysis indicates that the hyper-parameter of OAC has little impact on OAC-DE's performance.

## About this repository

### How to install

`pip install oacde`

### How to use

`import oacde`

### Tutorial

[Click here to read the full tutorial.](https://github.com/sustech-opal/oac-de/blob/main/tutorial.ipynb)


## Citation

1. You may cite this work in a scientific context as:

    Huang, C., Bai, H. & Yao, X. Online algorithm configuration for differential evolution algorithm. Appl Intell 52, 9193–9211 (2022), DOI: [10.1007/s10489-021-02752-1](https://doi.org/10.1007/s10489-021-02752-1)

2. Or copy the folloing BibTex file:

    ```latex
    @article{OACDE,
    author  = {Changwu Huang and Hao Bai and Xin Yao},
    title     = {Online algorithm configuration for differential evolution algorithm},
    journal   = {Applied Intelligence},
    volume    = {52},
    number    = {8},
    pages     = {9193--9211},
    year      = {2022},
    url       = {https://doi.org/10.1007/s10489-021-02752-1}
    doi       = {10.1007/s10489-021-02752-1},
    }
    ```

3. Or [download the citation in RIS file.](https://citation-needed.springer.com/v2/references/10.1007/s10489-021-02752-1?format=refman&flavour=citation)

## Contact

[Asst. Prof. Changwu HUANG](https://faculty.sustech.edu.cn/huangcw3/en/)
