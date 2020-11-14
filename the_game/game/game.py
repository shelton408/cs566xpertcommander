import numpy as np
import random
import logging

HANDSIZES = {
    '1': 8,
    '2': 7,
    '3': 6  # 6 cards on hand for 3 or more players
}


class Game:
    def __init__(self, num_players, num_cards=98, is_static_drawpile=False):
        self.minMoveSize = 2
        self.num_players = num_players
        self.num_cards = num_cards
        self.is_static_drawpile = is_static_drawpile

        if self.is_static_drawpile:
            self.static_drawpile = np.arange(2, self.num_cards + 2)
            np.random.shuffle(self.static_drawpile)
        else:
            self.static_drawpile = np.array([])

        self.handsize = HANDSIZES[str(3 if self.num_players > 2 else self.num_players)]
        self.state = {
            'current_player': 0,
            'handsize': self.handsize,
            'players': [],
            'num_moves_taken': 0,
            'number_of_cards': self.num_cards,
            'drawpile': [],
            'played_cards': np.array([], dtype=int),
            'decks': np.array([], dtype=int),
            'hands': [],
            'hints': [[], [], [], []],  # Not implemented yet
            'legal_actions': [[], [], [], []],
            'last_action': ()
        }


    def draw(self, num_cards):
        if num_cards < len(self.state['drawpile']):
            cards = self.state['drawpile'][:num_cards]
            self.state['drawpile'] = self.state['drawpile'][num_cards:]
        else:
            cards = self.state['drawpile']
            self.state['drawpile'] = np.array([])

            # when drawpile is empty, min moves lowered to 1
            self.minMoveSize = 1
        return np.array(cards)


    def init_game(self):
        handsize = HANDSIZES[str(3 if self.num_players > 2 else self.num_players)]
        self.state = {
            'current_player': 0,
            'handsize': self.handsize,
            'players': [],
            'num_moves_taken': 0,
            'number_of_cards': self.num_cards,
            'drawpile': [],
            'played_cards': np.array([], dtype=int),
            'decks': np.array([], dtype=int),
            'hands': [],
            'hints': [[], [], [], []],  # Not implemented yet
            'legal_actions': [[], [], [], []],
            'last_action': ()
        }
        self.state['players'] = list(range(self.num_players))

        self.state['decks'] = np.array([1, 1, self.num_cards + 2, self.num_cards + 2])  # First two decks are ascending, other are descending

        if not self.is_static_drawpile:
            drawpile = np.arange(2, self.num_cards + 2)
            np.random.shuffle(drawpile)
        else:
            drawpile = self.static_drawpile.copy()
        self.state['drawpile'] = drawpile

        handsize = HANDSIZES[str(3 if self.num_players > 2 else self.num_players)]
        for idx in range(self.num_players):
            hand = np.sort(self.draw(handsize))
            # self.state['hands'][idx] = np.array(hand)

            # self.state['hands'] = np.insert(self.state['hands'], hand, axis=1)
            self.state['hands'].append(hand)
            self.state['legal_actions'][idx] = self.get_legal_actions(idx)

        # Check who has the best starting hand
        # hand_evals = self.evaluate_hands()
        # self.state['current_player_id'] = hand_evals.index(max(hand_evals))
        # Can also use random
        self.state['current_player'] = random.randint(0, self.num_players - 1)
        return (self.state, self.state['current_player'])

    @staticmethod
    def _is_legal_move(deck_value, card_value, isAscending):
        if isAscending:
            return card_value > deck_value\
                or deck_value - 10 == card_value
        else:
            return card_value < deck_value\
                or deck_value + 10 == card_value

    def _can_be_played(self, card):
        for d_i, is_ascending in enumerate([True, True, False, False]):
            if self._is_legal_move(self.state['decks'][d_i], card, is_ascending):
                return True
        return False


    def get_all_legal_actions(self):
        for idx in range(self.num_players):
            self.state['legal_actions'][idx] = self.get_legal_actions(idx)


    def get_legal_actions(self, agent_id):
        legal_actions = []
        one_hot = []
        for c_id, c_value in enumerate(self.state['hands'][agent_id]):
            for d_id, d_value in enumerate(self.state['decks']):
                isAscending = True if d_id in [0, 1] else False
                if self._is_legal_move(d_value, c_value, isAscending):
                    one_hot.append(1)
                    legal_actions.append((c_id, d_id))
                else:
                    one_hot.append(0)
        while len(one_hot) < self.state['handsize']*4:
            one_hot.append(0)
        if self.state['num_moves_taken'] >= self.minMoveSize:
            legal_actions.append((-1, -1))  # Encode the end-turn action as negative value
            one_hot.append(1)
        else:
            one_hot.append(0)
        return one_hot

    def step(self, action_id):
        #action = self.state['legal_actions'][self.state['current_player']][action_id]
        if not self.state['legal_actions'][self.state['current_player']][action_id]:
            print('ILLEGAL MOVE CHOSEN')
            return (self.state, self.state['current_player'])
        else:
            action = (action_id//4, action_id % 4)
            if action_id == self.state['handsize']*4:
                action = (-1,-1)

            if action == (-1, -1):  # End of turn action
                new_cards = self.draw(self.state['num_moves_taken'])

                #not sure we should sort this
                #self.state['hands'][self.state['current_player']] = sorted(np.append(self.state['hands'][self.state['current_player']], new_cards))
                self.state['hands'][self.state['current_player']] = np.append(self.state['hands'][self.state['current_player']], new_cards)
                self.state['num_moves_taken'] = 0

                # Because players get removed from self.state['players'] in the endgame when they
                # have played all their cards.
                next_id = self.state['players'].index(self.state['current_player']) + 1
                self.state['current_player'] = self.state['players'][next_id] if next_id < self.num_players else 0
                self.get_all_legal_actions()
                self.state['last_action'] = action
                return (self.state, self.state['current_player'])

            else:
                card_id, deck_id = action
                card = self.state['hands'][self.state['current_player']][card_id]
                self.state['decks'][deck_id] = card
                self.state['played_cards'] = np.sort(np.append(self.state['played_cards'], card))

                self.state['hands'][self.state['current_player']] = np.delete(self.state['hands'][self.state['current_player']], card_id)
                self.state['num_moves_taken'] += 1
                self.get_all_legal_actions()
                self.state['last_action'] = action
                logging.info(' State for player {}: {}\nEvaluation: {}\nLast Card(s) played{}\n'.format(self.state['current_player'], str(self.state), str(self.drawpile_eval()), action))
                return (self.state, self.state['current_player'])


    def is_over(self):
        '''
        Two ways the game ends.
        1) The game is won when the drawpile is finished and every player has an empty hand.
        2) The game is over if the current player can not take any action.
        '''

        return (len(self.state['drawpile']) == 0 and all([len(hand) == 0 for hand in self.state['hands']]))\
            or sum(self.state['legal_actions'][self.state['current_player']]) == 0


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
