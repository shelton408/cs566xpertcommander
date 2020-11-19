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
        self.target = {'a1': Card('1'), 'a2': Card('1'), 'd1': Card('50'), 'd2': Card('50')}
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

        # draw a card
        self._perform_draw_action(players)

        # move to next player
        self.current_player = (self.current_player + self.direction) % self.num_players

    def get_legal_actions(self, players, player_id):
        legal_actions = []
        hand = players[player_id].hand
        for card in hand:
            if card > self.target['a1']:
                legal_actions.append('a1-'+card.rank)
            if card > self.target['a2']:
                legal_actions.append('a2-'+card.rank)
            if card < self.target['d1']:
                legal_actions.append('d1-'+card.rank)
            if card < self.target['d2']:
                legal_actions.append('d2-'+card.rank)
        if not legal_actions:
            self.is_over = True
        return legal_actions

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
        if not self.dealer.deck:
            self.is_over = True
            return None
        card = self.dealer.deck.pop()
        players[self.current_player].hand.append(card)
