import random


class RandomAgent:
    '''A Random Agent'''

    def __init__(self, id):
        self.id = id

    def step(self, state):
        if state['num_moves_taken'] == 2:
            # Return the final legal action because it is the END-TURN action
            return len(state['legal_actions'][self.id]) - 1
        return random.randint(0, len(state['legal_actions'][self.id]) - 1)
