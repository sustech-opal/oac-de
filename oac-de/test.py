#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' Copyright 2022 Changwu Huang, Hao Bai and Xin Yao

    [DEBUG ONLY] Validation of modules and functions
'''
import time
from problems import problem as pb
from publics import parameterize as para
from algorithms.algodev import OACDE


tik = time.time()
#! -----------------------------------------------------------------------------
seed = 0  # ! HB: don't change this value
# rng = np.random.RandomState(seed)
# list_seed = tuple(rng.randint(0, 2**31-1, 25))
case = "ALL"
outputs = []

## 1. Problem Configuration
benchmark = "bbob2015"
D = 20
funID = 7 # 8 9 Rosenbrock , 15 Rastrigin
problem = pb.Benchmark(benchmark_set=benchmark, D=D, funID=funID,
    instanceID=1)

## 2. Algorithm Configuration
NP = 24
config = para.DE(seed = seed,
                    N = NP,
                    mutation='de/rand/1'
                    )
stop = para.StopCondition(max_FES=100.*NP, max_iter=None, delta_ftarget=1e-8)

## 3. Aglorithm Selection: using DE random algorithm
if case == "OACDE" or case == 1 or case == "ALL":
    print("--- Testing OACDE:")
    optimizer = OACDE(opt_problem=problem,
                        algo_parameters=config,
                        stop_conditions=stop,
                        mut_pool=[
                        "de/rand/1", # CoDE, SaDE, EPSDE, SAKPDE, jDE
                        "de/rand/2", # CoDE, SaDE, SAKPDE
                        "de/best/2", # EPSDE, SAKPDE
                        "de/current-to-best/2",  # SaDE, SAKPDE
                        "de/current-to-rand/1",  # CoDE, SaDE, EPSDE
                        "de/current-to-pbest/1", # JADE, SHADE
                        ],
                        cx_pool=["bin", "exp"],
                        tuning_level="individual_level",
                        F_range=(0.0, 1.0),
                        CR_range=(0.0, 1.0),
                        model_utilization_strategy="success",
                        stagnation_gen=10,
                        KDE_kernel="gaussian",
                        KDE_width=1/3,
                        KDE_max_size=None,
                        MAB_algo="ucb",
                        MAB_value_method="sample_average",
                        epsilon=0.2,
                        temperature=None,
                        sliding_window_size=None, # required for
                                                # "sliding_window_average"
                        )
    results = optimizer.solve(disp=1)




#! -----------------------------------------------------------------------------
tok = time.time()

# 5. Post-processing
print("\n--- OK : Elapsed Time {:.2f}s ---".format(tok-tik))
print("Calculated results:", results)
print("Theoretical optimal value:", problem.f_opt_theory)
print("Relative error:", results[1]-problem.f_opt_theory)

print(optimizer.history)