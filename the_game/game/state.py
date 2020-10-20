import numpy as np


class Deck:
    def __init__(self, init_num, ascending=True):
        self.top_card = init_num
        self.isAscending = ascending
        self.isDecending = not ascending

    def __str__(self):
        return str(self.top_card)


class State:
    """
    The Game State class
    """

    def __init__(self, num_players):
        self.decks = [
            Deck(init_num, ascending=True) if init_num == 1
            else Deck(init_num, ascending=False)
            for init_num in [1, 1, 100, 100]]

        self.drawpile = np.arange(2, 100)
        np.random.shuffle(self.drawpile)

        # expert mode is 3
        # maybe worth looking into if project is going well
        self.minMoveSize = 2

        # Variable to indicate if the game state can proceed further.
        # If a player can't take his turn he will change the state to an EndState
        self.isEndState = False

    def __str__(self):
        return '''
        Decks: {} {} {} {}
        Num Cards in drawpile: {}
        '''.format(self.decks[0], self.decks[1], self.decks[2], self.decks[3], len(self.drawpile))


    def draw(self, num_cards):
        '''
        Function that pops N number of cards from the drawpile.
        :param num_cards: The number of cards to be drawn from the drawpile
        :return: List of top N cards from the drawpile.
        '''
        if num_cards < len(self.drawpile):
            cards = self.drawpile[:num_cards]
            self.drawpile = self.drawpile[num_cards:]
        else:
            cards = self.drawpile
            self.drawpile = np.array([])

            # when drawpile is empty, min moves lowered to 1
            self.minMoveSize = 1
        return cards


    # @staticmethod
    # def is_legal_move(card, public_card, ascending=True):
    #     """
    #     :param card: card in hand
    #     :param public_card: top card of deck
    #     :param ascending: If the deck containing the public_card must be ascending or descending
    #     :return:
    #     """
    #
    #     # if card is exactly 10 over or under it is valid
    #     if public_card + 10 == card or public_card - 10 == card:
    #         return 1
    #
    #     if ascending and card > public_card:
    #         return 1
    #     elif (not ascending) and card < public_card:
    #         return 1
    #     else:
    #         return 0


    def get_legal_moves(self, hand):
        """
        returns all possible moves based on hand and current deck state
        this does not modify the state in any way
        :param hand: 1D ndarray represents the player's hand
        :return: (N,D) ndarray mapping player hand (N) to possible moves for each deck (D)
        """

        # currently just going to iterate over each card in hand and each
        # card on a deck to determine possible moves
        # this seems like it might be computationally inefficient
        # so maybe upgrade later

        # an (N, D) ndarray where N is hand size and D is each deck
        # binary representation 0, 1 for each index of deck if it is a valid move
        legal_moves = np.zeros((hand.shape[0], len(self.decks)))

        # decks 0, 1 descending, 2, 3 ascending
        for card in range(len(hand)):
            for public_card in range(len(self.decks)):
                ascending = public_card > 1
                legal_moves[card, public_card] = self.is_legal_move(hand[card], self.decks[public_card], ascending)

        return legal_moves

    # for now, just a card and target deck
    # implement hints at some point
    # apply action and return new set of legal_moves based on that action
    # must take legal parameters (i.e. parameters must be checked to be legal before calling this method)
    def apply_action(self, hand, card, deck, moves):
        self.decks[deck] = hand[card]
        hand = np.delete(hand, card)
        moves = np.delete(moves, card, axis=0)
        for c in range(len(hand)):
            moves[c, deck] = self.is_legal_move(hand[c], self.decks[deck], deck > 1)

        return hand, moves


    def is_playable(self, hand):
        '''
        Boolean function that return True if there is any card in the
        hand that can be played on any deck
        '''
        for card in hand:
            for i in range(len(self.decks)):
                if self.is_legal_move(card, i):
                    return True
        print('NOT PLAYABLE')
        return False

    def play_card(self, card, deck_index):
        '''
        Change the top number of the specific deck to the card number
        :param card: The card number that is played
        :param deck_index: The index of the deck that we want to add on.
        :return: None
        '''
        self.decks[deck_index].top_card = card


    def is_legal_move(self, card, deck_index):
        '''
        Checks if the card can be played on the deck
        :param card: The integer value for the card.
        :param deck_index: The index of the deck that we want to add on.
        :return: True if the move is valid, else False
        '''
        if self.decks[deck_index].isAscending:
            return card > self.decks[deck_index].top_card\
                or self.decks[deck_index].top_card - 10 == card
        else:
            return card < self.decks[deck_index].top_card\
                or self.decks[deck_index].top_card + 10 == card
