#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' [DEBUG ONLY] Validation of modules and functions
'''
import time
import numpy as np
from problems import problem as pb
from operators import DEmat as DE
RS = np.random.RandomState(0)



tik = time.time()
#! -----------------------------------------------------------------------------
##* Problem setting
D = 100
# benchmark = "cec2017"
benchmark = "bbob2015"
F = 1
seed = 111
problem = pb.Benchmark(benchmark_set=benchmark,
                       D=D, funID=F, instanceID=1)

#! -----------------------------------------------------------------------------
NP = 100
MaxGen = 100
MaxFES = int(NP*MaxGen)
f_error = 1e-8

##* initiate population
LB = np.array([[-100,]*D,])
UB = np.array([[100,]*D,])

pop = RS.uniform(LB, UB, (NP, D))
# print(pop, pop.shape)

# evaluate 0 gen
F = problem.fobj(pop)
# print("F:", F, F.shape)

g = NP

pop_archive = []

#! -----------------------------------------------------------------------------
while True:
    ##* mutate
    # pop_mutant = DE.rand_1(pop, 0.3, RS)
    # pop_mutant = DE.rand_2(pop, 0.3, RS)
    # pop_mutant = DE.best_1(pop, 0.3, RS, fitness=F)
    # pop_mutant = DE.best_2(pop, 0.3, RS, fitness=F)
    # pop_mutant = DE.rand_to_best_1(pop, 0.3, RS, fitness=F)
    # pop_mutant = DE.rand_to_pbest_1(pop, 0.3, RS, fitness=F, p=0.2)
    # pop_mutant = DE.current_1(pop, 0.3, RS)
    # pop_mutant = DE.current_to_rand_1(pop, 0.3, RS)
    # pop_mutant = DE.current_to_best_1(pop, 0.3, RS, fitness=F)
    pop_mutant = DE.current_to_best_2(pop, 0.5, RS, fitness=F)
    # pop_mutant = DE.current_to_pbest_1(pop, 0.3, RS, fitness=F, p=0.2, 
    #     pop_archive=pop_archive)
    # print("pop_mutant:", pop_mutant, pop_mutant.shape)
    

    ##* crossover
    # pop_trial = pop_mutant
    pop_trial = DE.bin(pop, pop_mutant, 0.5, RS)
    # pop_trial = DE.exp(pop, pop_mutant, 0.5, RS)
    # pop_trial = DE.none(pop, pop_mutant)
    # pop_trial = DE.eig(pop, pop_mutant, 0.5, RS)
    # print("pop_trial:", pop_trial, pop_trial.shape)

    ##* evaluate
    F_trial = problem.fobj(pop_trial)
    # print("F trial:", F_trial, F_trial.shape)


    ##* select
    pop, F = DE.select(pop, F, pop_trial, F_trial)
    

    ##* stop criteria
    min_idx = np.argmin(F)
    f_opt = F[min_idx,]
    hat_f_err = f_opt - problem.f_opt_theory
    if hat_f_err <= f_error:
        # x_opt = np.where(pop == f_opt) #? why it works (reserved for Hao)
        x_opt = pop[min_idx,] # , for advanced indexing
        print("--- SUCCESS ---")
        print("x_opt: ", x_opt)
        print("f_opt: ", f_opt)
        break
    # print("g = {}: f_opt = {}, hat_f_err = {}".format(g, f_opt, hat_f_err))

    ##* some extra operations
    pop_archive.append(pop[min_idx, ]) # only for current_to_pbest_1
    if g >= MaxFES:
        break
    else:
        g = g + NP


#! -----------------------------------------------------------------------------
tok = time.time()
print("g =", g, "f_opt =", F.min())
print("--- OK : {}s ---".format(tok-tik))
