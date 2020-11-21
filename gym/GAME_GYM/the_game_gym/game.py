import numpy as np
import random


HANDSIZES = {
    '1': 8,  # 8 cards in hand for a single player
    '2': 7,  # 7 cards in hand for a two-player game
    '3': 6,  # 6 cards in hand for 3 of more players
}

ILLEGAL_ACTION_REWARD = -1
ACTION_REWARD = 1
END_TURN_REWARD = 2


class Game:
    def __init__(self, num_players, num_cards=98, is_static=False):
        self.minMoveSize = 2
        self.num_players = num_players
        self.num_cards = num_cards
        self.is_static = is_static

        # If static we don't randomize between games
        if self.is_static:
            self.static_drawpile = np.arange(2, self.num_cards + 2, dtype=int)
            np.random.shuffle(self.static_drawpile)
        else:
            self.static_drawpile = np.array([])

        self.handsize = HANDSIZES[str(3 if self.num_players > 2 else self.num_players)]
        self.reset()

    def reset(self):
        '''
        Resets the state according to settings
        Returns: A Dict:State for a new game
        '''
        self.minMoveSize = 2
        state = {
            'handsize': self.handsize,
            'num_moves_taken': 0,
            'number_of_cards': self.num_cards,
            'hands': [],
            'played_cards': np.array([], dtype=int),
            'unplayed_cards': np.arange(2, self.num_cards + 2, dtype=int),
            'hints': [[0] * 4] * self.num_players,  # Hints given by players prioritising the decks
            'legal_actions': [[]] * self.num_players,
            'last_action': ()
        }

        state['players'] = list(range(self.num_players))

        # First two decks are ascending, other are descending
        state['decks'] = np.array([1, 1, self.num_cards + 2, self.num_cards + 2])


        # If game is set to static then all games begin with the same drawpile and same player to start
        if self.is_static:
            drawpile = self.static_drawpile.copy()
            state['current_player'] = 0
        else:
            drawpile = np.arange(2, self.num_cards + 2)
            np.random.shuffle(drawpile)
            state['current_player'] = random.randint(0, self.num_players - 1)

        # loop over players and draw hands
        for idx in range(self.num_players):
            hand, drawpile = self.draw(self.handsize, drawpile)
            state['hands'].append(hand)

            state['legal_actions'][idx] = self.get_legal_actions(idx, state)
        state['drawpile'] = drawpile
        self.state = state

    def draw(self, num_cards, drawpile):
        '''
        Takes in a number of cards to draw and drawpile and returns a Hand:np.array and
        the new drawpile after hand has been drawn
        '''

        if num_cards < len(drawpile):
            cards = drawpile[:num_cards]
            drawpile = drawpile[num_cards:]
        else:
            cards = drawpile
            drawpile = np.array([])

            # when drawpile is empty, min moves lowered to 1
            self.minMoveSize = 1
        return np.array(cards), drawpile


    @staticmethod
    def _is_legal_move(deck_value, card_value, isAscending):

        # Card value if 0 to fill up handsize when number of playable
        # cards in hand is less than self.handsize, so it's not a real card
        if card_value == 0:
            return False

        if isAscending:
            return card_value > deck_value\
                or deck_value - 10 == card_value
        else:
            return card_value < deck_value\
                or deck_value + 10 == card_value

    def get_legal_actions(self, player_id, state):
        # A one hot encoded vector of size self.handsize * 4
        one_hot = []

        # loop throuch card in hand
        for c_id, c_value in enumerate(state['hands'][player_id]):
            # loop through decks
            for d_id, d_value in enumerate(state['decks']):
                # only the first two decks are ascending
                isAscending = True if d_id in [0, 1] else False
                if c_value and self._is_legal_move(d_value, c_value, isAscending):
                    one_hot.append(1)
                else:
                    one_hot.append(0)

        if state['num_moves_taken'] >= self.minMoveSize:
            one_hot.append(1)
        else:
            one_hot.append(0)
        return np.array(one_hot)

    def get_all_legal_actions(self):
        for idx in range(self.num_players):
            self.state['legal_actions'][idx] = self.get_legal_actions(idx, self.state)


    def _pad_hand(self, hand):
        '''
        Makes sure that hand is always of size self.handsize
        Pads with zeros otherwise
        '''
        full_hand = np.zeros(self.handsize, dtype=int)
        length = min(self.handsize, len(hand))
        full_hand[-length:] = hand[-length:]
        return full_hand

    def step(self, action) -> (dict, int, bool):
        # Illegal action picked
        if self.state['legal_actions'][self.state['current_player']][action] == 0:
            # print('Illegal_Action')
            return self.state, ILLEGAL_ACTION_REWARD, False

        # last action available is END-OF-TURN action
        if action == (self.handsize * 4):

            # If there are still cards in drawpile:
            # Draw number of cards equal to moves taken
            if len(self.state['drawpile']) > 0:
                new_cards, self.state['drawpile'] = self.draw(self.state['num_moves_taken'], self.state['drawpile'])
                new_hand = np.sort(np.append(self.state['hands'][self.state['current_player']], new_cards))
                self.state['hands'][self.state['current_player']] = new_hand

            # Make sure that hand is always of size self.handsize
            self.state['hands'][self.state['current_player']] = self._pad_hand(
                self.state['hands'][self.state['current_player']])


            # Because players get removed from self.state['players'] in the endgame when they
            # have played all their cards.
            next_id = self.state['players'].index(self.state['current_player']) + 1
            self.state['current_player'] = self.state['players'][next_id] if next_id < self.num_players else 0
            self.state['num_moves_taken'] = 0
            self.get_all_legal_actions()
            self.state['last_action'] = 'END TURN'


            return self.state, END_TURN_REWARD, True

        # Card played
        else:
            # Because action is derived from a one hot encoded of size
            # self.handsize * 4
            card_id, deck_id = (action // 4, action % 4)

            card = self.state['hands'][self.state['current_player']][card_id]
            self.state['decks'][deck_id] = card
            self.state['played_cards'] = np.sort(np.append(self.state['played_cards'], card))
            self.state['unplayed_cards'] = np.delete(self.state['unplayed_cards'], np.where(self.state['unplayed_cards'] == card))

            # Delete played card from hand and pad hand
            self.state['hands'][self.state['current_player']] = np.delete(self.state['hands'][self.state['current_player']], card_id)
            self.state['hands'][self.state['current_player']] = self._pad_hand(
                self.state['hands'][self.state['current_player']])

            self.state['num_moves_taken'] += 1
            self.get_all_legal_actions()

            deck_desc = 'asc deck' if deck_id in [0, 1] else 'desc deck'
            self.state['last_action'] = '{} on {}:{}'.format(card, deck_desc, deck_id)
            return self.state, ACTION_REWARD, True


    def is_over(self):
        '''
        Two ways the game ends.
        1) The game is won when the drawpile is finished and every player has a hand full of 0 (empty hand).
        2) The game is over if the current player can not take any action.
        '''

        return (len(self.state['drawpile']) == 0 and all([sum(hand) == 0 for hand in self.state['hands']]))\
            or sum(self.state['legal_actions'][self.state['current_player']]) == 0


    def _can_be_played(self, card):
        for d_i, is_ascending in enumerate([True, True, False, False]):
            if self._is_legal_move(self.state['decks'][d_i], card, is_ascending):
                return True
        return False

    def hand_eval(self):
        '''
        Evaluation on how many cards in player hands can be played
        Returns: (number of cards in all hands that can be played) / (total number of cards in players hands)
        '''
        playable_cards = 0
        total_cards = 0
        for hand in self.state['hands']:
            for c in hand:
                if self._can_be_played(c):
                    playable_cards += 1
                total_cards += 1
        return playable_cards / total_cards


    def drawpile_eval(self):
        '''
        Evaluation function that returns the ratio on how many cards in the drawpile can be played.
        Returns: (number of cards in drawpile that can be played) / (number of cards in drawpile)
        '''
        if len(self.state['drawpile']) > 0:
            return len([c for c in self.state['drawpile'] if self._can_be_played(c)]) / len(self.state['drawpile'])
        else:
            return 0  # drawpile is empty

    def num_playable(self):
        '''
        returns raw # of cards that can be played
        '''
        return len([c for c in self.state['drawpile'] if self._can_be_played(c)])

    def num_unplayed_playable(self):
        '''
        Returns number of cards that are unplayed that can be played
        '''
        return len([c for c in self.state['unplayed_cards'] if self._can_be_played(c)])


    def eval(self):
        '''
        returns the sum of number of decks an unplayed card can be played upon
        '''
        eval = 0
        for card in self.state['unplayed_cards']:
            for d_i, is_ascending in enumerate([True, True, False, False]):
                if self._is_legal_move(self.state['decks'][d_i], card, is_ascending):
                    eval += 1

        return eval / (len(self.state['unplayed_cards']) * 4)
