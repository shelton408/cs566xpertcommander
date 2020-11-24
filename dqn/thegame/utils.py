import os
import json
import numpy as np
from collections import OrderedDict

import rlcard

from rlcard.games.thegame.card import TheGameCard as Card

# Read required docs
ROOT_PATH = rlcard.__path__[0]

# a map of abstract action to its index and a list of abstract action
with open(os.path.join(ROOT_PATH, 'games/thegame/jsondata/action_space.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)
    ACTION_LIST = list(ACTION_SPACE.keys())

def init_deck(deck_size=98):
    ''' Initialize a standard deck of cards

    Args:
        deck_size (int): number of cards in the deck

    Returns:
        (list): A list of Card object
    '''
    rank_list = [str(i) for i in range(2, deck_size+2)]
    res = [Card(rank) for rank in rank_list]
    return res

def encode_hand(plane, hand):
    ''' Encode hand and represerve it into plane
    Args:
        plane (np array):  numpy array
        hand (list): list of string of hand's card
    Returns:
        (array):  numpy array
    '''
    for index, card in enumerate(hand):
        # card_index = int(card) - 1
        # plane[card_index] = 1
        plane[index] = int(card) / 100.0  # normalize to [0, 1]
    plane.sort()

def encode_target(plane, targets):
    ''' Encode target and represerve it into plane

    Args:
        plane (np.array): numpy array
        targets(str): list of Cards from decks

    Returns:
        (array): numpy array
    '''
    for index, target in enumerate(targets):
        # target_index = int(target) - 1
        # plane[index, target_index] = 1
        plane[index] = int(target) / 100
    return plane

def encode_card(plane, cards):
    ''' Encode hand and represerve it into plane

    Args:
        plane (np.array):  numpy array
        cards (list): list of string of hand's card

    Returns:
        (array):  numpy array
    '''
    for card in cards:
        # cards go from 2-99 so indices 0-97 represent them
        card_index = int(card) - 2
        plane[card_index] = 1
    return plane

def cards2list(cards):
    ''' Get the corresponding string representation of cards

    Args:
        cards (list): list of UnoCards objects

    Returns:
        (string): string representation of cards
    '''
    cards_list = []
    for card in cards:
        cards_list.append(str(card))
    return cards_list

def targets2list(targets):
    ''' Get the corresponding string representation of cards

    Args:
        targets (dict): dict of the 4 decks

    Returns:
        (string): string representation of cards
    '''
    targets_list = [str(targets['a1']), str(targets['a2']), str(targets['d1']), str(targets['d2'])]

    return targets_list
