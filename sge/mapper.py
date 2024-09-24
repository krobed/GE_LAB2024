from sge.parameters import (
    params,
    set_parameters,
    load_parameters
)
from collections import deque
import sge.grammar_pge as grammar_pge
import copy

def mapper_GE(genome):
    genome = list(genome)
    """
    A fast genotype to phenotype mapping process. Map input via rules to
    output. Does not require the recursive tree class, but still calculates
    tree information, e.g. number of nodes and maximum depth.

    :param genome: A genome to be mapped.
    :return: Output in the form of a phenotype string ('None' if invalid),
             Genome,
             None (this is reserved for the derivation tree),
             The number of nodes in the derivation,
             A boolean flag for whether or not the individual is invalid,
             The maximum depth of any node in the tree, and
             The number of used codons.
    """

    # Create local variables to avoid multiple dictionary lookups
    max_tree_depth, max_wraps = params['MAX_TREE_DEPTH'], params['MAX_WRAPS']
    bnf_grammar = params['BNF_GRAMMAR']

    n_input = len(genome)

    # Depth, max_depth, and nodes start from 1 to account for starting root
    # Initialise number of wraps at -1 (since
    used_input, current_depth, max_depth, nodes, wraps = 0, 1, 1, 1, -1

    # Initialise output as empty deque list (deque is a list-like container
    # with fast appends and pops on either end).
    output = deque()

    # Initialise the list of unexpanded non-terminals with the start rule.
    unexpanded_symbols = deque([(bnf_grammar.start_rule, 1)])

    while (wraps < max_wraps) and unexpanded_symbols:
        # While there are unexpanded non-terminals, and we are below our
        # wrapping limit, we can continue to map the genome.

        if max_tree_depth and (max_depth > max_tree_depth):
            # We have breached our maximum tree depth limit.
            break

        if used_input % n_input == 0 and \
                used_input > 0 and \
                any([i[0]["type"] == "NT" for i in unexpanded_symbols]):
            # If we have reached the end of the genome and unexpanded
            # non-terminals remain, then we need to wrap back to the start
            # of the genome again. Can break the while loop.
            wraps += 1

        # Expand a production from the list of unexpanded non-terminals.
        current_item = unexpanded_symbols.popleft()
        current_symbol, current_depth = current_item[0], current_item[1]

        if max_depth < current_depth:
            # Set the new maximum depth.
            max_depth = current_depth

        # Set output if it is a terminal.
        if current_symbol["type"] != "NT":
            output.append(current_symbol["symbol"])

        else:
            # Current item is a new non-terminal. Find associated production
            # choices.
            production_choices = bnf_grammar.rules[current_symbol[
                "symbol"]]["choices"]
            no_choices = bnf_grammar.rules[current_symbol["symbol"]][
                "no_choices"]

            # Select a production based on the next available codon in the
            # genome.
            current_production = genome[used_input % n_input] % no_choices

            # Use an input
            used_input += 1

            # Initialise children as empty deque list.
            children = deque()
            nt_count = 0

            for prod in production_choices[current_production]['choice']:
                # iterate over all elements of chosen production rule.

                child = [prod, current_depth + 1]

                # Extendleft reverses the order, thus reverse adding.
                children.appendleft(child)
                if child[0]["type"] == "NT":
                    nt_count += 1

            # Add the new children to the list of unexpanded symbols.
            unexpanded_symbols.extendleft(children)

            if nt_count > 0:
                nodes += nt_count
            else:
                nodes += 1

    # Generate phenotype string.
    output = "".join(output)
    if bnf_grammar.file_name.endswith("pybnf"):
        output = bnf_grammar.python_filter(output)

    if len(unexpanded_symbols) > 0:
        # All non-terminals have not been completely expanded, invalid
        # solution.
        return None, genome, None, nodes, True, max_depth, used_input

    return output, genome, None, nodes, False, max_depth, used_input


def probabilistic_mapping(codon, productions):
    """ Probabilistic mapping, using a PCFG."""
    idx_selected_rule = len(productions) - 1
    prob_aux = 0.0
    for i in range(len(productions)):
        prob_aux += productions[i][1]
        if codon < prob_aux:
            idx_selected_rule = i
            break
    return idx_selected_rule

def mapper_PGE(genotype):
    """ Genotype-phenotype mapping function. Returns the phenotype and a counter.
    The counter stores the number of times each production rule was expanded. 
    The counter is later used on the function to update the PCFG's probabilities."""
    start = grammar_pge.start_rule()
    phenotype = [start]

    ind_pointer = 0
    wraps = -1
    pos = 0
    gram_counter = copy.deepcopy(grammar_pge.get_counter())
    while wraps < params['MAX_WRAPS']:        
        codon = genotype[pos]
        symbol = phenotype.pop(ind_pointer)
        productions = grammar_pge.get_dict()[symbol]     # get rules from symbol NT

        idx_selected_rule = probabilistic_mapping(codon, productions)

        gram_counter[symbol][idx_selected_rule] += 1

        # append at the beggining of the list
        p = ind_pointer
        for prod in productions[idx_selected_rule][0]:
            phenotype.insert(p,prod[0])    # append selected production
            p += 1
        if grammar_pge.is_individual_t(phenotype):
            break
        else:
            for _ in range(ind_pointer, len(phenotype)):
                if phenotype[ind_pointer] in grammar_pge.get_non_terminals():
                    break
                else:
                    ind_pointer += 1
        pos += 1
        if pos >= len(genotype):
            pos = 0
            wraps += 1
    return "".join(phenotype), gram_counter