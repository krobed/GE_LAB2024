import random
from numpy import cos, sin
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from sge.engine import setup
import sge
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from multiprocessing import Process, Queue
import time



class SymbolicRegression():
    def __init__(self, has_test_set=False, invalid_fitness=np.inf):
        self.__invalid_fitness = invalid_fitness
        self.read_fit_cases()

    def read_fit_cases(self):
        self.df = pd.read_csv('resources/GRID/LIB/df_cdrag_53.txt',sep=',').sample(n=1000,random_state=1)
        self.X = self.df.values[:,:-1]
        self.Y = self.df.values[:,-1]

    def get_error(self, individual, Y_train, dataset, queue):
        try:
            Y_pred = list(map(lambda x: eval(individual), dataset))
            error = mean_squared_error(Y_train,Y_pred)
        except Exception as e: 
            error = self.__invalid_fitness
        if error==None:
            error = self.__invalid_fitness
        queue.put(error)


    def get_error_time(self, individual, Y_train, dataset):
        q = Queue()
        p = Process(target=self.get_error, args=(individual, Y_train, dataset, q))
        max_time = 10
        t0 = time.time()
        p.start()
        while time.time() - t0 < max_time:
            p.join(timeout=1)
            if not p.is_alive():
                break
        if p.is_alive():
            #process didn't finish in time so we terminate it
            print('El proceso se estancó')
            p.terminate()
            result = np.inf
        else:
            result = q.get()
        return result


    def evaluate(self, individual):
        if individual is None:
            return self.__invalid_fitness
        fitness_train = self.get_error_time(individual, self.Y, self.X)
        fitness_val = np.inf
        return fitness_train,fitness_val, {'generation': 0, "evals": 1, "test_error": fitness_val}

if __name__ == "__main__":
    import sge
    eval_func = SymbolicRegression()
    sge.evolutionary_algorithm(evaluation_function=eval_func)