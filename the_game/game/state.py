import numpy as np


class State():
    '''
    The Game State class
    '''

    def __init__(self, num_players):
        self.decks = np.array([100, 100, 1, 1])
        self.drawpile = np.arange(2, 100)
        np.random.shuffle(self.drawpile)
        self.minMoveSize = 2

    def __str__(self):
        return '''
        Decks: {}
        Cards in drawpile: {}
        '''.format(self.decks, len(self.drawpile))


    def draw(self, num_cards):
        if num_cards >= len(self.drawpile):
            cards = self.drawpile[:num_cards]
            self.drawpile = self.drawpile[num_cards:]
        else:
            cards = self.drawpile
            self.drawpile = np.array([])
        return cards

    def eval(self):
        print('evaluate the state')

    def apply_action(self, action):
        print('return new state when action has been applied')
