import random
from numpy import cos, sin
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from sge.engine import setup
import sge
import argparse
import numpy as np

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


class SymbolicRegression():
    def __init__(self, has_test_set=False, invalid_fitness=9999999):
        self.__train_set = []
        self.__test_set = None
        self.__number_of_variables = 1
        self.__invalid_fitness = invalid_fitness
        self.partition_rng = random.Random()
        self.has_test_set = has_test_set
        self.readpolynomial()
        self.calculate_rrse_denominators()
        
    def calculate_rrse_denominators(self):
        self.__RRSE_train_denominator = 0
        self.__RRSE_test_denominator = 0
        train_outputs = [entry[-1] for entry in self.__train_set]
        train_output_mean = float(sum(train_outputs)) / len(train_outputs)
        self.__RRSE_train_denominator = sum([(i - train_output_mean)**2 for i in train_outputs])
        if self.__test_set:
            test_outputs = [entry[-1] for entry in self.__test_set]
            test_output_mean = float(sum(test_outputs)) / len(test_outputs)
            self.__RRSE_test_denominator = sum([(i - test_output_mean)**2 for i in test_outputs])

    def read_fit_cases(self):
        f_in = open(self.__file_problem,'r')
        data = f_in.readlines()
        f_in.close()
        fit_cases_str = [ case[:-1].split() for case in data[1:]]
        self.__train_set = [[float(elem) for elem in case] for case in fit_cases_str]
        self.__number_of_variables = len(self.__train_set[0]) - 1

    def readpolynomial(self):
        def Koza_3(inp):
            return pow(inp,6) + pow(inp,5) + pow(inp,4) + pow(inp,3) + pow(inp,2)  + inp

        self.function = Koza_3
        # two variables
        l = []
        for xx in np.linspace(-1, 1, 20):
            zz = self.function(xx)
            l.append([xx,zz])

        self.__train_set=l
        self.training_set_size = len(self.__train_set)

    def get_error(self, individual, dataset):
        pred_error = 0
        for fit_case in dataset:
            case_output = fit_case[-1]
            try:
                result = eval(individual, globals(), {"x": fit_case[:-1]})
                pred_error += (case_output - result)**2
            except (OverflowError, ValueError) as e:
                return self.__invalid_fitness
        return pred_error

    def evaluate(self, individual):
        error = 0.0
        test_error = 0.0
        if individual is None:
            return None

        error = self.get_error(individual, self.__train_set)
        error = _sqrt_( error /self.__RRSE_train_denominator)

        if error is None:
            error = self.__invalid_fitness

        if self.__test_set is not None:
            test_error = 0
            test_error = self.get_error(individual, self.__test_set)
            test_error = _sqrt_( test_error / float(self.__RRSE_test_denominator))

        return error, {'generation': 0, "evals": 1, "test_error": test_error}


if __name__ == "__main__":
    import sge
    eval_func = SymbolicRegression()
    sge.evolutionary_algorithm(evaluation_function=eval_func)