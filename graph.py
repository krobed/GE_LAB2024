<<<<<<< HEAD
=======

>>>>>>> 97ae1a2a8db90a78d3d93e90229bdbd79d5cec69
import numpy as np

import pandas as pd
import os

import matplotlib.pyplot as plt
from sge.parameters import params
import seaborn as sns

from sge.engine import setup

path_grid_csv = 'grid_parameters.csv'
path = 'results/n/'

def get_info_best_fitness(forder):
  runs_array = os.listdir(forder)
  best_fitness_array = []
  for run in runs_array:
      run_path = forder  + run + '/'
      df = pd.read_csv(run_path + 'progress_report.csv', names=['fitness', 'mean', 'std', 'valids'])
      best_fitness_array.append(list(df['fitness'])[-1])
  mean_best_fitness = np.mean(best_fitness_array)
  std_best_fitness = np.std(best_fitness_array)
  return mean_best_fitness,std_best_fitness

def get_columnas_variable(df):
    resultado = df.nunique()>1
    columnas_valiables = list(resultado.index[resultado.values])
    return columnas_valiables

def get_best_fitness(df,path):
    mean_best_fitness_list = []
    std_best_fitness_list = []
    for index, row in df.iterrows():
        path_folder = path + str(index) + '/'
        mean_best_fitness,std_best_fitness = get_info_best_fitness(path_folder)
        mean_best_fitness_list.append(mean_best_fitness)
        std_best_fitness_list.append(std_best_fitness)
    df['mean best fitness'] = mean_best_fitness_list
    df['std best fitness'] = std_best_fitness_list
    return df


df_grid = pd.read_csv(path_grid_csv)

columnas_valiables = get_columnas_variable(df_grid)


df_grid = get_best_fitness(df_grid,path)

df_grid_pivot = df_grid.pivot(columnas_valiables[0],columnas_valiables[1], 'mean best fitness')


fig, axis = plt.subplots(1,1, figsize=(10,8))
sns.heatmap(df_grid_pivot,ax=axis)
plt.title('Error en obtener un expresión para número de Nusselt en función \n del tamaño de población y la cantidad de generaciones',fontsize=16)
<<<<<<< HEAD
plt.xlabel('Adaptive increment (proportional)',fontsize=14)
plt.ylabel('Learning Factor',fontsize=14)
=======
plt.xlabel('Tamaño de población',fontsize=14)
plt.ylabel('Cantidad de generaciones',fontsize=14)
>>>>>>> 97ae1a2a8db90a78d3d93e90229bdbd79d5cec69
plt.savefig('graph.png')