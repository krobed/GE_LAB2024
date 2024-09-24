import random
import sge.grammar_sge as grammar_sge
import copy
import numpy as np
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)

def crossover(p1, p2):
    if params['ALGORITHM']=='SGE':
        return crossover_sge(p1, p2)
    elif params['ALGORITHM']=='GE' or params['ALGORITHM']=='PGE':
        return crossover_ge(p1, p2)

def crossover_sge(p1, p2):
    xover_p_value = 0.5
    gen_size = len(p1['genotype'])
    mask = [random.random() for i in range(gen_size)]
    genotype = []
    for index, prob in enumerate(mask):
        if prob < xover_p_value:
            genotype.append(p1['genotype'][index][:])
        else:
            genotype.append(p2['genotype'][index][:])
    mapping_values = [0] * gen_size
    # compute nem individual
    _, tree_depth = grammar_sge.mapping(genotype, mapping_values)
    return {'genotype': genotype, 'fitness': None, 'mapping_values': mapping_values, 'tree_depth': tree_depth}


def crossover_ge(p1, p2, nr_cuts = 2):
    """ Crossover function. """
    genotype = copy.deepcopy(p1['genotype'])
    gen_size = len(p1['genotype'])
    cuts = []
    while len(cuts) != nr_cuts:
        cutPoint = np.random.randint(0,gen_size)
        if cutPoint not in cuts:
            cuts.append(cutPoint)
    cuts.sort()
    partnerGenotype = False
    start = 0
    for i in range(0, nr_cuts):
        if i != 0:
            start = cuts[i-1]
        partnerGenotype = not partnerGenotype
        if partnerGenotype:
            genotype[start:cuts[i]] = p2['genotype'][start:cuts[i]]

    return {'genotype': genotype, 'fitness': None}