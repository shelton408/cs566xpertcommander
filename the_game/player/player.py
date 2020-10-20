import numpy as np


class Player:
    """
    Human player class.
    """

    def __init__(self, number, hand):
        self.hand = hand
        self.number = number

    def __str__(self):
        return 'Player {} w. hand {}'.format(str(self.number), self.hand)

    def can_play(self):
        return len(self.hand) > 0

    def take_turn(self, state):
        '''
        Player takes it turn by playing at least 'minMoveSize' number of times.
        If that isn't possible then make state an endState and return it.
        Otherwise ask player for action and apply it to the state.
        :param state: The current state that players action will be applied to.
        :return: The next state after changes have been applied.
        '''
        actions_taken = 0
        is_turn_over = False
        while not is_turn_over:

            if not state.is_playable(self.hand):
                if actions_taken < state.minMoveSize:
                    state.isEndState = True
                return state


            print('Player {}\'s turn'.format(str(self.number)))
            print(self.hand)
            print(state)
            print('enter index of card followed by index of deck to play it on')
            if actions_taken >= state.minMoveSize:
                print('or enter \'end\' to end the turn')

            move = input()

            # if ending turn draw the same number of cards as played
            if actions_taken >= state.minMoveSize and move == 'end':
                self.hand = np.append(self.hand, state.draw(actions_taken))
                is_turn_over = True
            else:
                try:
                    card_index, deck_index = map(int, move.split(' '))
                except ValueError:
                    print('Not a valid input')
                    continue

                if state.is_legal_move(self.hand[card_index], deck_index):
                    state.play_card(self.hand[card_index], deck_index)
                    self.hand = np.delete(self.hand, card_index)
                    actions_taken += 1
                else:
                    # maybe make this more descriptive at some point
                    print('that is an invalid move')

        return state
