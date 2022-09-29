# -*- coding: utf-8 -*-
import numpy as np
def select(pop, F, pop_trial, F_trial):
    ''' Compare target population and trial population and select survivals

        Return
        ------
        pop : 2D-array
            survival population
        F : 1D-array
            survival fitness corresponding to the survival population
    '''
    NP, D = pop.shape
    flag = np.array(F_trial <= F).reshape(-1, NP)
    cond = np.tile(np.transpose(flag), (1, D))
    pop = np.where(cond, pop_trial, pop)
    flag = np.squeeze(flag)
    F = np.where(flag, F_trial, F)
    return pop, F, flag #TODO needs to adapt the last output to all algos
