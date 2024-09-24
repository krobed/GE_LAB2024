import copy
import random
import sge.grammar_sge as grammar_sge
import copy
import numpy as np
from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)
from random import randint, shuffle

def mutate(p, pmutation):
    if params['ALGORITHM']=='SGE':
        return mutate_sge(p, pmutation)
    elif params['ALGORITHM']=='GE' or params['ALGORITHM']=='PGE':
        return mutate_ge(p, pmutation)

def mutate_sge(p, pmutation):
    p = copy.deepcopy(p)
    p['fitness'] = None
    size_of_genes = grammar_sge.count_number_of_options_in_production()
    mutable_genes = [index for index, nt in enumerate(grammar_sge.get_non_terminals()) if size_of_genes[nt] != 1 and len(p['genotype'][index]) > 0]
    for at_gene in mutable_genes:
        nt = list(grammar_sge.get_non_terminals())[at_gene]
        temp = p['mapping_values']
        mapped = temp[at_gene]
        for position_to_mutate in range(0, mapped):
            if random.random() < pmutation:
                current_value = p['genotype'][at_gene][position_to_mutate]
                choices = []
                if p['tree_depth'] >= grammar_sge.get_max_depth():
                    choices = grammar_sge.get_non_recursive_options()[nt]
                else:
                    choices = list(range(0, size_of_genes[nt]))
                    choices.remove(current_value)
                if len(choices) == 0:
                    choices = range(0, size_of_genes[nt])
                p['genotype'][at_gene][position_to_mutate] = random.choice(choices)
    return p

def mutate_ge(ind, prob_mutation = 0.9):
    """ Mutation function."""
    gen_size = len(ind['genotype'])
    new_gen = copy.deepcopy(ind['genotype'])
    for i in range(gen_size):
        if np.random.uniform() < prob_mutation:
            new_gen[i] = randint(0, params['CODON_SIZE'])
    
    return {'genotype': new_gen, 'fitness': None}