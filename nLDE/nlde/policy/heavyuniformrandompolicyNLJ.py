"""
Created on Nov 24, 2011

@author: Chang Qin, Maribel Acosta
"""
from multiprocessing import Manager
import random
class OperatorRate(object):

    def __init__(self, left, right, output):
        self.left = left
        self.right = right
        self.output = output

class HeavyRandomPolicyNLJ(object):
    """
    Implements a routing policy that goes through the loop and calculates the selectivity, but the final decision is always made randomly,
    disregarding the selectivity. So that make a light-weight random policy also heavy.
    """

    def __init__(self):
        manager = Manager()
        self.priority_table = manager.dict()
        self.initial_plan = manager.dict()

    def initialize_priorities(self, plan_order):

        self.initial_plan = plan_order

        for operator in plan_order.keys():
            self.priority_table.update({operator: OperatorRate(0.0, 0.0, 0.0)})

    def select_operator(self, operators, operators_desc, tup=None, operators_vars=None, operators_not_sym=[]):

        selected_operator = -1
        highest_priority = -2
        all_output_zero = True

        operators_selectable = list(set(operators))

        if len(operators_selectable) == 1:
            return operators_selectable[0]

        if len(operators_selectable) == 0:
            return operators[0]

        for operator in operators_selectable:

            input_card = self.priority_table[operator].left + self.priority_table[operator].right

            # Case: 'operators' has received no input,
            # route to first operators according to plan.
            if input_card == 0:
                return operator

            # Check whether 'operators' has produced output.
            if self.priority_table[operator].output == 0:
                inverse_selectivity = -1
            else:
                inverse_selectivity = input_card / self.priority_table[operator].output
                all_output_zero = False

            if inverse_selectivity > highest_priority:
                selected_operator = operator
                highest_priority = inverse_selectivity

        i = random.randint(0, len(operators_selectable) - 1)
        selected_operator = operators_selectable[i]

        return selected_operator

    def update_priorities(self, tup, queue=-1):
        pass
