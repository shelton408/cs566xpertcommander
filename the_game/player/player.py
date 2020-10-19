import numpy as np


class Player:
    """
    Human player class.
    """

    def __init__(self, hand):
        self.hand = hand

    def take_turn(self, state):
        print('take the turn')
        legal_moves = state.get_legal_moves(self.hand)

        # pick a legal move, play it reevaluate turn
        # maybe find some method of only reevaluating decks which have changed based on previous action
        # or have the take_action method return a set of new legal moves
        # need some way to decide when done with turn, maybe (N,D+1) where last index is ending turn or something
