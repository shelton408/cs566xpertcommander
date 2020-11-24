from rlcard.games.thegame.card import TheGameCard as Card
from rlcard.games.thegame.utils import cards2list, targets2list

class TheGameRound(object):

    def __init__(self, dealer, num_players, np_random):
        ''' Initialize the round class

        Args:
            dealer (object): the object of UnoDealer
            num_players (int): the number of players in game
        '''
        self.np_random = np_random
        self.dealer = dealer
        self.target = {'a1': Card('1'), 'a2': Card('1'), 'd1': Card('100'), 'd2': Card('100')}
        self.current_player = 0
        self.num_players = num_players
        self.direction = 1
        self.played_cards = []
        self.is_over = False

    def proceed_round(self, players, action):
        ''' Call other Classes's functions to keep one round running

        Args:
            player (object): object of TheGamePlayer
            action (str): string of legal action
        '''
        player = players[self.current_player]

        if action == 'draw':
            # draw cards
            self._perform_draw_action(players)

        else:
            action_info = action.split('-')
            pile = action_info[0]
            rank = action_info[1]

            # remove corresponding card
            remove_index = None
            for index, card in enumerate(player.hand):
                if rank == card.rank:
                    remove_index = index
                    break
            card = player.hand.pop(remove_index)
            self.played_cards.append(card)

            # update target
            self.target[pile] = card

        # move to next player
        self.current_player = (self.current_player + self.direction) % self.num_players

    def get_legal_actions(self, players, player_id):
        legal_actions = []
        hand = players[player_id].hand
        for card in hand:
            # asending or +-10
            if card > self.target['a1'] or card.diff_ten(self.target['a1']):
                legal_actions.append('a1-'+card.rank)
            if card > self.target['a2'] or card.diff_ten(self.target['a2']):
                legal_actions.append('a2-'+card.rank)
            # descending or +-10
            if card < self.target['d1'] or card.diff_ten(self.target['d1']):
                legal_actions.append('d1-'+card.rank)
            if card < self.target['d2'] or card.diff_ten(self.target['d2']):
                legal_actions.append('d2-'+card.rank)

        if len(hand) <= 6:
            legal_actions.append('draw')

        if not legal_actions:
            self.is_over = True
        return legal_actions

    def get_playable_cards(self):

        num_playable_cards = 0
        deck = self.dealer.deck

        for card in deck:
            # asending or +-10
            if card > self.target['a1'] or card.diff_ten(self.target['a1']):
                num_playable_cards += 1
            elif card > self.target['a2'] or card.diff_ten(self.target['a2']):
                num_playable_cards += 1
            # descending or +-10
            elif card < self.target['d1'] or card.diff_ten(self.target['d1']):
                num_playable_cards += 1
            elif card < self.target['d2'] or card.diff_ten(self.target['d2']):
                num_playable_cards += 1

        return num_playable_cards

    def get_state(self, players, player_id):
        ''' Get player's state

        Args:
            players (list): The list of TheGamePlayer
            player_id (int): The id of the player
        '''
        state = {}
        player = players[player_id]
        state['hand'] = cards2list(player.hand)
        state['target'] = targets2list(self.target) # a1, a2, d1, d2
        state['playable_cards'] = cards2list(self.dealer.deck)
        state['played_cards'] = cards2list(self.played_cards)
        state['legal_actions'] = self.get_legal_actions(players, player_id)
        return state

    def _perform_draw_action(self, players):

        # num of cards
        n_cards = 8 - len(players[self.current_player].hand)

        if len(self.dealer.deck) < n_cards:
            # draw all cards
            n_cards = len(self.dealer.deck)

        for _ in range(n_cards):
            card = self.dealer.deck.pop()
            players[self.current_player].hand.append(card)
