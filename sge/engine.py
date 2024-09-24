from ast import Continue
import random
import sys
from token import OP
import numpy as np
import pandas as pd
import sge.grammar_sge as grammar_sge
import sge.grammar_pge as grammar_pge
import sge.grammar_ge as grammar_ge
import sge.logger as logger
from datetime import datetime
from tqdm import tqdm
from sge.operators.recombination import crossover
from sge.operators.mutation import mutate
from sge.operators.selection import tournament
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)
from random import randint, shuffle
from sge.mapper import mapper_GE, mapper_PGE
from sge.functions_pge import update_probs
import copy

import re
from scipy.optimize import minimize
from multiprocessing import Process, Queue
import time

from sge.utilities.stats.trackers import cache


def generate_random_individual():
    genotype, tree_depth = None, None
    if params['ALGORITHM']=='SGE':
        genotype = [[] for key in grammar_sge.get_non_terminals()]
        tree_depth = grammar_sge.recursive_individual_creation(genotype, grammar_sge.start_rule()[0], 0)
    elif params['ALGORITHM']=='GE':
        genotype = [randint(0, params['CODON_SIZE']) for _ in range(params['SIZE_GENOTYPE'])]
    elif params['ALGORITHM']=='PGE':
        genotype = [np.random.uniform() for _ in range(params['SIZE_GENOTYPE'])]
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth}

def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()

def evaluate(ind, eval_func, OPTIMIZE=False):
    inf_info = list(ind.keys())
    phen, tree_depth, other_info, quality, quality_val,opt_const = None, None, None, np.inf, np.inf, []
    if pd.isna(ind['fitness']):
        if params['ALGORITHM']=='SGE':
            #print('mapping')
            mapping_values = [0 for i in ind['genotype']]
            ind['original_phenotype'], tree_depth = grammar_sge.mapping(ind['genotype'], mapping_values)
            ind['mapping_values'] = mapping_values
        elif params['ALGORITHM']=='GE':
            ind['original_phenotype'], genome, tree, nodes, invalid, tree_depth, used_codonsmapper = mapper_GE(ind['genotype'])
        elif params['ALGORITHM']=='PGE':
            ind['original_phenotype'], ind['gram_counter'] = mapper_PGE(ind['genotype'])
        if "Constant" in ind['original_phenotype']:
            phen,opt_const = Get_phtnotype_time(ind['original_phenotype'],[],eval_func, OPTIMIZE)
        else:
            phen = ind['original_phenotype']
        if (params['CACHE']  and (phen not in cache.keys())) or not params['CACHE']:
            try:
                quality, quality_val, other_info = eval_func.evaluate(phen)
            except:
                pass
        if pd.isna(quality):
            quality = np.inf
        if params['CACHE']:
            cache[phen] = quality
        ind['phenotype'] = phen
        ind['opt_const'] = opt_const
        ind['fitness'] = quality
        ind['fitness val'] = quality_val
        ind['other_info'] = other_info
        ind['tree_depth'] = tree_depth
        ind['optimized'] = OPTIMIZE
    elif (not ind['optimized']) and OPTIMIZE and ("Constant" in ind['original_phenotype']):    
        if 'opt_const' in inf_info: 
            phen,opt_const = Get_phtnotype_time(ind['original_phenotype'],ind['opt_const'],eval_func, OPTIMIZE)
        else:
            phen,opt_const = Get_phtnotype_time(ind['original_phenotype'],[],eval_func, OPTIMIZE)
        if phen not in cache.keys():
            try:
                quality,quality_val, other_info = eval_func.evaluate(phen)
            except:
                pass
            if quality == None:
                quality = np.inf
            if params['CACHE']:
                cache[phen] = quality
        ind['opt_const'] = opt_const
        ind['phenotype'] = phen
        ind['fitness val'] = quality_val
        ind['fitness'] = quality
        ind['other_info'] = other_info
        ind['optimized'] = OPTIMIZE
    return ind

def Get_phtnotype_time(phenotype, old_constants, fitness_function, OPTIMIZE):
    def f(phenotype, old_constants, fitness_function, OPTIMIZE, queue):
        res = Get_phenotype(phenotype, old_constants, fitness_function, OPTIMIZE)
        queue.put(res)

    q = Queue()
    p = Process(target=f, args=(phenotype, old_constants, fitness_function, OPTIMIZE, q))
    max_time = 30
    t0 = time.time()

    p.start()
    while time.time() - t0 < max_time:
        p.join(timeout=1)
        if not p.is_alive():
            break

    if p.is_alive():
        #process didn't finish in time so we terminate it
        p.terminate()
        replace_phenotype, opt_const = Get_phenotype(phenotype, old_constants, fitness_function, False)
    else:
        replace_phenotype, opt_const = q.get()
    return replace_phenotype, opt_const

def Get_phenotype(phenotype, old_constants, fitness_function, OPTIMIZE):
    p = r"Constant"
    n_constants = len(re.findall(p, phenotype))

    replace_phenotype = phenotype
    for i in range(n_constants):
      replace_phenotype = replace_phenotype.replace('Constant', 'c[' + str(i) + ']',1)

    def eval_ind(c):
        aux = replace_phenotype
        for i in range(len(c)):
            aux = aux.replace('c[' + str(i) + ']', str(c[i]))
        return fitness_function.evaluate(aux)[0]

    if n_constants>0:
        if len(old_constants)==0:
            old_constants = np.random.rand(n_constants)
        if OPTIMIZE:
            try:
                fun = lambda x: eval_ind(x)
                res = minimize(fun, old_constants, method='SLSQP',jac=False)
                opt_const = res['x']
            except:
                opt_const = old_constants
        else:
            opt_const = old_constants
        for index in range(n_constants):
            replace_phenotype = replace_phenotype.replace('c[' + str(index) + ']', str(opt_const[index]))
    return replace_phenotype, opt_const

def setup(parameters_file_path = None):
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
    set_parameters(sys.argv[1:])
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    logger.prepare_dumps()
    random.seed(params['SEED'])
    if params['ALGORITHM']=='SGE':
        grammar_sge.set_path(params['GRAMMAR'])
        grammar_sge.read_grammar()
        grammar_sge.set_max_tree_depth(params['MAX_TREE_DEPTH'])
        grammar_sge.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])
    elif params['ALGORITHM']=='GE':
        params['BNF_GRAMMAR'] = grammar_ge.grammar_ge(params['GRAMMAR'])
    elif params['ALGORITHM']=='PGE':
        grammar_pge.set_path(params['GRAMMAR'])
        grammar_pge.read_grammar()

def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    #sys.stdout.write("\r INICIALIZANDO                                             ")
    setup(parameters_file_path=parameters_file)
    population = list(make_initial_population())
    it = 0
    best_overall = {}
    flag = False
    while it <= params['GENERATIONS']:
        #print('########### Generation ' + str(it) + ' ########')
        #sys.stdout.write("\r Generation " + str(it) + '                                 ') 
        if params['CACHE'] and it%params['CLEAN_CACHE_EACH']==0 and it!=0:
            cache = {}
        for i in range(len((population))):
            #sys.stdout.write("\r Generation " + str(it) + ': evaluando individuo ' +  str(i) + '                   ') 
            if params['OPTIMIZE'] and it%params['OPTIMIZE_EACH'] == 0 and it!=0:
                population[i] = evaluate(population[i], evaluation_function,OPTIMIZE=True)
            else:
                population[i] = evaluate(population[i], evaluation_function,OPTIMIZE=False)
            if params['ALL_VALID']:
                while (pd.isna(population[i]['fitness'])) or population[i]['fitness']>10**6:
                    population[i] = generate_random_individual()
                    population[i] = evaluate(population[i], evaluation_function,OPTIMIZE=False)        

        population.sort(key=lambda x: x['fitness'])

        if params['ALGORITHM']=='PGE':
            if population[0]['fitness'] <= best_overall.setdefault('fitness', np.inf):
                best_overall = copy.deepcopy(population[0])
            if not flag:
                update_probs(best_overall, params['LEARNING_FACTOR'])
            else:
                update_probs(best_generation, params['LEARNING_FACTOR'])
            flag = not flag

        logger.evolution_progress(it, population)
        new_population = population[:params['ELITISM']]
        while len(new_population) < params['POPSIZE']:
            #sys.stdout.write("\r Generation " + str(it) + ': crossover                     ') 
            if random.random() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            #sys.stdout.write("\r Generation " + str(it) + ': mutation                     ') 
            ni = mutate(ni, params['PROB_MUTATION'])
            new_population.append(ni)

        population = new_population

        if params['ALGORITHM']=='PGE':
            best_generation = copy.deepcopy(new_population[0])
            if params['ADAPTIVE']:
                params['LEARNING_FACTOR'] += params['ADAPTIVE_INCREMENT']
        it += 1

