import random
from numpy import cos, sin
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from sge.engine import setup
import sge
import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error


class SymbolicRegression():
    def __init__(self, has_test_set=False, invalid_fitness=9999999):
        self.__invalid_fitness = invalid_fitness
        self.read_fit_cases()

    def read_fit_cases(self):
        df = pd.read_csv('resources/LIB/TF/tf.txt',sep=',')
        self.X_train = df.values[:,:-1]
        self.Y_train = df.values[:,-1]

    def get_error(self, individual, dataset):
        try:
            Y_pred = list(map(lambda x: eval(individual), dataset))
            error = mean_squared_error(self.Y_train,Y_pred,squared=False)
        except:
            return self.__invalid_fitness
        return error

    def evaluate(self, individual):
        if individual is None:
            return self.__invalid_fitness
        error = self.get_error(individual, self.X_train)
        if error is None:
            error = self.__invalid_fitness
        return error, {'generation': 0, "evals": 1}

if __name__ == "__main__":
    import sge
    eval_func = SymbolicRegression()
    sge.evolutionary_algorithm(evaluation_function=eval_func)