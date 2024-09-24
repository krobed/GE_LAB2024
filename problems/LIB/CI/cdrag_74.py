import random
from numpy import cos, sin
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from sge.engine import setup
import sge
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


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
        #print(individual)
        try:
            Y_pred = list(map(lambda x: eval(individual), dataset))
            error = mean_squared_error(Y_train,Y_pred, squared=False)
        except Exception as e: 
            return self.__invalid_fitness
        if error==None:
            return self.__invalid_fitness
        return error

    def evaluate(self, individual):
        if individual is None:
            return self.__invalid_fitness
        error_25 = self.get_error(individual, self.Y_25, self.X_25)
        error_53 = self.get_error(individual, self.Y_53, self.X_53)
        error_74 = self.get_error(individual, self.Y_74, self.X_74)
        error_102 = self.get_error(individual, self.Y_102, self.X_102)
        fitness_train = error_74
        fitness_val = np.mean([error_53,error_25,error_102])
        return fitness_train,fitness_val, {'fitness 25': error_25, 'fitness 53': error_53,'fitness 74': error_74,'fitness 102': error_102}

if __name__ == "__main__":
    import sge
    eval_func = SymbolicRegression()
    sge.evolutionary_algorithm(evaluation_function=eval_func)