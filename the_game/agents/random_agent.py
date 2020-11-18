import random


class RandomAgent:
    '''A Random Agent'''

    def __init__(self, id):
        self.id = id

    def step(self, state):
        legal_actions = state['legal_actions'][self.id] 
        actions = [i for i in range(len(legal_actions)) if legal_actions[i] != 0]
        if state['num_moves_taken'] == 2 and legal_actions[len(legal_actions) - 1]:
            # Return the final legal action because it is the END-TURN action
            return len(legal_actions) - 1
        return random.choice(actions)
