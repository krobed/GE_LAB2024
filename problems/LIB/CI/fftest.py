import random
from numpy import cos, sin
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
from sge.engine import setup
import sge
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import re
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import autograd.numpy as anp
import warnings
warnings.filterwarnings('ignore')

class SymbolicRegression():
    def __init__(self, has_test_set=False, invalid_fitness=9999999):
        self.__invalid_fitness = invalid_fitness
        self.read_fit_cases()

    def read_fit_cases(self):
        self.df_25 = pd.read_csv('resources/LIB/CI/df_ff_25.txt',sep=',').sample(n=1000, random_state=1)
        self.X_25 = self.df_25.values[:,:-1]
        self.Y_25 = self.df_25.values[:,-1]
        self.df_53 = pd.read_csv('resources/LIB/CI/df_ff_53.txt',sep=',').sample(n=1000, random_state=1)
        self.X_53 = self.df_53.values[:,:-1]
        self.Y_53 = self.df_53.values[:,-1]
        self.df_74 = pd.read_csv('resources/LIB/CI/df_ff_74.txt',sep=',').sample(n=1000, random_state=1)
        self.X_74 = self.df_74.values[:,:-1]
        self.Y_74 = self.df_74.values[:,-1]
        self.df_102 = pd.read_csv('resources/LIB/CI/df_ff_102.txt',sep=',').sample(n=1000, random_state=1)
        self.X_102 = self.df_102.values[:,:-1]
        self.Y_102 = self.df_102.values[:,-1]

    def get_error(self, individual, Y_train, dataset):
        #print(individual)
        try:
            Y_pred = list(map(lambda x: eval(individual), dataset))
            error = root_mean_squared_error(Y_train,Y_pred)
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
        fitness_train = np.mean([error_53,error_74,error_25])
        fitness_val = error_102
        return fitness_train,fitness_val, {'fitness 25': error_25, 'fitness 53': error_53,'fitness 74': error_74,'fitness 102': error_102}


eval_func = SymbolicRegression()
phen, tree_depth, other_info, quality, quality_val,opt_const = 'Constant*(x[0]**Constant )*(x[1]**Constant)', None, None, np.inf, np.inf, []
ind = [[0],[0],[0,2],[3,2,3],[2,2],[2,2],[1,0,0],[1,0]]
mapping_values = [0 for i in ind]
# phenotype, tree_depth = grammar_sge.mapping(ind, mapping_values)
p = r"Constant"
n_constants = len(re.findall(p, phen))

replace_phenotype = phen
for i in range(n_constants):
    replace_phenotype = replace_phenotype.replace('Constant', 'c[' + str(i) + ']',1)

# Definir el problema
class MyProblem(Problem):
    def __init__(self, n_var = 3, xl= [], xu = []):
        super().__init__(
            n_var=n_var,  # Número de variables (constantes a optimizar)
            n_obj=1,  # Número de objetivos
            n_constr=0,  # Número de restricciones
            xl=xl,  # Límites inferiores
            xu=xu,  # Límites superiores
        )

    def _evaluate(self, x, out,*args, **kwargs):
        # Evalúa tu ecuación (por ejemplo, 'a*x1 + b*x2') con eval
        def eval_ind(c):
            aux = replace_phenotype
            for i in range(len(c)):
                aux = aux.replace('c[' + str(i) + ']', str(c[i]))
            return eval_func.evaluate(aux)[0]
        f= np.zeros(len(x))
        for i in range(len(x)):
            e = eval_ind(x[i])
            f[i]=e  # Evalúa el string con las variables de x
        len(f)
        out["F"] =  f

# Instanciar el problema
problem = MyProblem(n_var = n_constants, xl = [-100]*n_constants, xu=[100]*n_constants)

# Configurar el algoritmo PSO
algorithm = PSO(
    pop_size=50,  # Tamaño de la población
    w=0.9,  # Factor de inercia
    c1=0.5,  # Componente cognitiva
    c2=0.3,  # Componente social
)


# Ejecutar la optimización
result = minimize(
    problem,
    algorithm,
    termination=("n_gen", 500),  # Número de generaciones
    seed=42,  # Para reproducibilidad
    verbose=True,
)

# Mostrar los resultados
print("Mejor solución encontrada:", result.X)
print("Valor de la función objetivo:", result.F)
