import string
import numpy as np
import sge
import pandas as pd
import os
import json
import itertools
import shutil
import subprocess
import sys
import argparse

from sge.parameters import params

from sge.engine import setup

def createFolder(folder):
    try:
        shutil.rmtree(mainPath + '/' + folder)
    except:
        pass
    try:
        os.makedirs(mainPath + '/' + folder)
    except:
        pass

# Se obtiene el path relativo
mainPath = os.path.abspath("")

df_grid = pd.read_csv(os.sep.join([mainPath, "grid.csv"]))

createFolder('results/')

for index, row in df_grid.iterrows():
    problem = str(row['problem'])
    parameters = str(row['parameters'])
    runs = row['runs']
    for algorithm in ['GE', 'SGE', 'PGE']:
        for i in range(runs):
            subprocess.check_call(['python','-m', 'problems.' + problem, '--experiment_name','results/'+ problem + '/' + algorithm, '--parameters', 'parameters/' + parameters, '--algorithm', algorithm, '--run', str(i)])

subprocess.check_call(['python','graph.py'])