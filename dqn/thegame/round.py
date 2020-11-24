from rlcard.games.thegame.utils import cards2list, targets2list
from rlcard.games.thegame.card import TheGameCard as Card

class TheGameRound(object):

    def __init__(self, dealer, num_players, np_random, deck_size=98):
        ''' Initialize the round class

        Args:
            dealer (object): the object of UnoDealer
            num_players (int): the number of players in game
        '''

        d_card = str(deck_size + 2)
        self.np_random = np_random
        self.dealer = dealer
        self.target = {'a1': Card('1'), 'a2': Card('1'), 'd1': Card(d_card), 'd2': Card(d_card)}
        self.current_player = 0
        self.num_players = num_players
        self.direction = 1
        self.played_cards = []
        self.is_over = False
        self.is_win = False
        self.played_turn_cards = []
        self.min_move_size = 2

    def proceed_round(self, players, action):
        ''' Call other Classes's functions to keep one round running

        Args:
            players (list): list of Player
            action (str): string of legal action
        '''

        player = players[self.current_player]
        if not action == 'pass':
            action_info = action.split('-')
            pile = action_info[0]
            rank = action_info[1]

            # remove corresponding card
            remove_index = player.hand.index(rank)
            card = player.hand.pop(remove_index)
            # more memory efficient to store as string
            self.played_cards.append(str(card))
            self.played_turn_cards.append(card)
            # update target
            self.target[pile] = card

        # player has ended their turn
        # draw cards and clear cards played this turn
        else:
            # draw a card
            # if deck is empty lower min move requirement
            self.dealer.deal_cards(player, num=len(self.played_turn_cards))
            if not self.dealer.deck:
                self.min_move_size = 1

                # check if any players have cards left
                # if not the game was a win
                player_cards_remaining = False
                for player in players:
                    if len(player.hand) > 0:
                        player_cards_remaining = True
                self.is_over = not player_cards_remaining
                self.is_win = self.is_over

            self.played_turn_cards = []
            self.current_player = (self.current_player + self.direction) % self.num_players

    def get_legal_actions(self, players, player_id):
        legal_actions = []
        hand = players[player_id].hand

        # if card hasn't already been played then check which deck in can go to
        # -1 is a placeholder used for when the deck has been emptied
        for card in hand:
            if card not in self.played_turn_cards:
                if card > self.target['a1'] or card == (int(self.target['a1']) - 10):
                    legal_actions.append('a1-' + card.rank)
                if card > self.target['a2'] or card == (int(self.target['a2']) - 10):
                    legal_actions.append('a2-' + card.rank)
                if card < self.target['d1'] or card == (int(self.target['d1']) + 10):
                    legal_actions.append('d1-' + card.rank)
                if card < self.target['d2'] or card == (int(self.target['d2']) + 10):
                    legal_actions.append('d2-' + card.rank)

        # if there are no legal actions, the min amount of cards haven't been played
        # and there are still cards in the deck, the game has been lost
        if not legal_actions \
                and len(self.played_turn_cards) < self.min_move_size \
                and self.dealer.deck:
            self.is_over = True

        # if the player has played min number of cards, or does not have a hand, they may pass
        if len(self.played_turn_cards) >= self.min_move_size \
                or not hand:
            legal_actions.append('pass')

        return legal_actions

    def get_state(self, players, player_id):
        ''' Get player's state

        Args:
            players (list): The list of Player
            player_id (int): The id of the player
        '''
        state = {}
        player = players[player_id]
        state['hand'] = cards2list(player.hand)
        state['target'] = targets2list(self.target) # a1, a2, d1, d2
        state['played_cards'] = cards2list(self.played_cards)
        state['legal_actions'] = self.get_legal_actions(players, player_id)
        state['turn_cards'] = self.played_turn_cards
        return state
