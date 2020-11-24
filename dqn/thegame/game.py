from copy import deepcopy
import numpy as np

from rlcard.games.thegame import Dealer
from rlcard.games.thegame import Player
from rlcard.games.thegame import Round

HANDSIZES = {
    '1': 8,
    '2': 7,
    '3': 6  # 6 cards on hand for 3 or more players
}

class TheGame(object):

    def __init__(self, num_players=1, deck_size=98, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = num_players
        self.deck_size = deck_size

    def init_game(self):
        ''' Initialize players and state

        Returns:
            (tuple): Tuple containing:

                (dict): The first state in one game
                (int): Current player's id
        '''
        # Initalize payoffs
        self.payoffs = [0 for _ in range(self.num_players)]

        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize one players to play the game
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]

        # deal hands based on num of players
        self.hand_size = HANDSIZES[str(3 if self.num_players > 2 else self.num_players)]
        for player in self.players:
            self.dealer.deal_cards(player, self.hand_size)

        # Initialize a Round
        self.round = Round(self.dealer, self.num_players, self.np_random)

        # Save the hisory for stepping back to the last state.
        self.history = []

        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''

        if self.allow_step_back:
            # First snapshot the current state
            his_dealer = deepcopy(self.dealer)
            his_round = deepcopy(self.round)
            his_players = deepcopy(self.players)
            self.history.append((his_dealer, his_players, his_round))

        self.round.proceed_round(self.players, action)
        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if not self.history:
            return False
        self.dealer, self.players, self.round = self.history.pop()
        return True

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        state = self.round.get_state(self.players, player_id)
        state['player_num'] = self.get_player_num()
        state['current_player'] = self.round.current_player
        return state

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        num_played_cards = len(self.round.played_cards)
        self.payoffs[0] = [1 for i in range(num_played_cards-1)] + [-1]
        return self.payoffs

    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''

        return self.round.get_legal_actions(self.players, self.round.current_player)

    def get_player_num(self):
        ''' Return the number of players in Limit Texas Hold'em

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    def get_action_num(self):
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions.
        '''

        # 4 decks and + 1 for pass action
        return 4 * self.deck_size + 1

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.round.current_player

    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        return self.round.is_over
