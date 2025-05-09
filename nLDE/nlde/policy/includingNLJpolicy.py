"""
Created on Nov 24, 2011

@author: Maribel Acosta
"""
import random
from policy import Policy
from multiprocessing import Manager

class IncludingNLJ(Policy):
    def __init__(self):
        self.priority_table = {}

    def initialize_priorities(self, plan_order):
        self.priority_table = {operator: 0 for operator in plan_order}
        #print("Initial order of operators:", list(plan_order))

    def select_operator(self, operators, operators_desc, tup=None, operators_vars=None, operators_not_sym=[]):
        #print("Initial order of operators passed to select_operator:", operators)

        # For NLJNotEnd policy, we want all operators at the beginning of the routing policy.
        candidates = list(set(operators))
        #print("Order of candidates after conversion to set and back to list:", candidates)

        # List of selectable operators containing non-cartesian product routes.
        operators_selectable = []
        for o in candidates:
            if set(tup.sources) & set(operators_desc[o].keys()):
                operators_selectable.append(o)
        #print("Order of operators_selectable:", operators_selectable)

        # Choose the operator with the least output so far
        selected_op = operators_selectable[0]
        min_output = self.priority_table[selected_op]

        for operator in operators_selectable:
            if self.priority_table[operator] < min_output:
                selected_op = operator
                min_output = self.priority_table[operator]

        #print("Selected operator:", selected_op)
        return selected_op

    def update_priorities(self, tup, queue=-1):
        if tup.from_operator is not None:
            if tup.from_operator not in self.priority_table:
                self.priority_table[tup.from_operator] = 0
            self.priority_table[tup.from_operator] += 1



    """
    def update_priorities(self, tup, queue=-1):
        #Track the number of tuples produced by each operator in the query plan
        if tup.from_operator is not None:
            self.priority_table[tup.from_operator] += 1
    """