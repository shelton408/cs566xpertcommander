import numpy as np


class Player():
    '''
    Human player class.
    '''

    def __init__(self):
        self.hand = np.array([])

    def take_turn(self, state):
        print('take the turn')
