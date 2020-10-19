import numpy as np


class Player:
    """
    Human player class.
    """

    def __init__(self, hand, isHuman=False):
        self.hand = hand
        self.isHuman = isHuman

    def take_turn(self, state):
        legal_moves = state.get_legal_moves(self.hand)
        actions_taken = 0
        while True:
            if self.isHuman:
                print('take the turn')
                print(self.hand)
                print(state)
                print('enter index of card followed by index of deck to play it on')
                if actions_taken >= state.minMoveSize:
                    print('or enter end to end turn')
                move = input()

                # if ending turn draw number of cards played
                if actions_taken >= state.minMoveSize:
                    if move == 'end':
                        self.hand = np.append(self.hand, state.draw(actions_taken))
                        break

                try:
                    card, deck = map(int, move.split(' '))
                except:
                    print('that is an invalid move')
                    continue

                if legal_moves[card, deck]:
                    self.hand, legal_moves = state.apply_action(self.hand, card, deck, legal_moves)
                    actions_taken += 1
                else:
                    # maybe make this more descriptive at some point
                    print('that is an invalid move')

        # pick a legal move, play it reevaluate turn
        # maybe find some method of only reevaluating decks which have changed based on previous action
        # or have the take_action method return a set of new legal moves
        # need some way to decide when done with turn, maybe (N,D+1) where last index is ending turn or something
