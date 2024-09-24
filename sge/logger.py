import numpy as np
from sge.parameters import params
import json
import os
import pandas as pd


def evolution_progress(generation, pop):
    fitness_samples = np.asarray([i['fitness'] for i in pop])
    data = '%4d,%.6e,%.6e,%.6e,%4d' % (generation, np.min(fitness_samples), np.mean(fitness_samples), np.std(fitness_samples), 100*np.sum(fitness_samples!=np.inf)/params['POPSIZE'])
    if params['VERBOSE']:
        print(data)
    save_progress_to_file(data)
    if generation % params['SAVE_STEP'] == 0:
        save_step(generation, pop)


def save_progress_to_file(data):
    with open('%s/run_%d/progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')

def save_step(generation, population):
    df = pd.DataFrame(population)
    df.to_csv('%s/run_%d/iteration_%d.csv' % (params['EXPERIMENT_NAME'], params['RUN'], generation))

def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    c = json.dumps(params_lower)
    open('%s/run_%d/parameters.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'a').write(c)


def prepare_dumps():
    try:
        os.makedirs('%s/run_%d' % (params['EXPERIMENT_NAME'], params['RUN']))
    except FileExistsError as e:
        pass
    save_parameters()