import random
from numpy import cos, sin
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from sge.engine import setup
import sge.grammar_sge as grammar_sge
import sge
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from scipy.optimize import minimize
import autograd.numpy as anp
import warnings
warnings.filterwarnings('ignore')

import sge
import re



class SymbolicRegression():
    def __init__(self, has_test_set=False, invalid_fitness=9999999):
        self.__invalid_fitness = invalid_fitness
        self.read_fit_cases()

    def read_fit_cases(self):
        self.df_25 = pd.read_csv('resources/LIB/CI/df_cdrag_25.txt',sep=',').sample(n=1000, random_state=1)
        self.X_25 = self.df_25.values[:,:-1]
        self.Y_25 = self.df_25.values[:,-1]
        self.df_53 = pd.read_csv('resources/LIB/CI/df_cdrag_53.txt',sep=',').sample(n=1000, random_state=1)
        self.X_53 = self.df_53.values[:,:-1]
        self.Y_53 = self.df_53.values[:,-1]
        self.df_74 = pd.read_csv('resources/LIB/CI/df_cdrag_74.txt',sep=',').sample(n=1000, random_state=1)
        self.X_74 = self.df_74.values[:,:-1]
        self.Y_74 = self.df_74.values[:,-1]
        self.df_102 = pd.read_csv('resources/LIB/CI/df_cdrag_102.txt',sep=',').sample(n=1000, random_state=1)
        self.X_102 = self.df_102.values[:,:-1]
        self.Y_102 = self.df_102.values[:,-1]

    def get_error(self, individual, Y_train, dataset):
        try:
            Y_pred = list(map(lambda x: eval(individual), dataset))
            error = root_mean_squared_error(Y_train,Y_pred)
        except Exception as e: 
            error= self.__invalid_fitness
        if error==None:
            error = self.__invalid_fitness
        return error

    def evaluate(self, individual):
        if individual is None:
            return self.__invalid_fitness
        error_25 = self.get_error(individual, self.Y_25, self.X_25)
        error_53 = self.get_error(individual, self.Y_53, self.X_53)
        error_74 = self.get_error(individual, self.Y_74, self.X_74)
        error_102 = self.get_error(individual, self.Y_102, self.X_102)
        fitness_train = np.mean([error_25,error_102,error_74])
        fitness_val = error_53
        return fitness_train,fitness_val, {'fitness 25': error_25, 'fitness 53': error_53,'fitness 74': error_74,'fitness 102': error_102}


eval_func = SymbolicRegression()
p = r"Constant"
aux = 'x[0]**Constant + Constant*x[1]**Constant'
n_constants = len(re.findall(p, aux))

replace_phenotype = aux
for i in range(n_constants):
    replace_phenotype = replace_phenotype.replace('Constant', 'c[' + str(i) + ']',1)
def eval_ind(c):
    aux = replace_phenotype
    for i in range(len(c)):
        aux = aux.replace('c[' + str(i) + ']', str(c[i]))
    return eval_func.evaluate(aux)[0]
old_constants = np.random.rand(n_constants)

fun = lambda x: eval_ind(x)
res = minimize(fun, old_constants, method='COBYQA')
opt_const = res['x']
print(opt_const)



