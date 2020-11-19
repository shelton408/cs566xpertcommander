import os
import json
import numpy as np
from collections import OrderedDict

import rlcard

from rlcard.games.thegame.card import TheGameCard as Card

# Read required docs
ROOT_PATH = rlcard.__path__[0]

# a map of abstract action to its index and a list of abstract action
with open(os.path.join(ROOT_PATH, 'games/thegame/jsondata/action_space_50.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)
    ACTION_LIST = list(ACTION_SPACE.keys())

def init_deck():
    ''' Initialize a standard deck of 98 cards

    Returns:
        (list): A list of Card object
    '''
    rank_list = Card.rank
    res = [Card(rank) for rank in rank_list]
    return res

def encode_card(plane, cards):
    ''' Encode hand and represerve it into plane

    Args:
        plane (array):  numpy array
        hand (list): list of string of hand's card

    Returns:
        (array):  numpy array
    '''
    for card in cards:
        card_index = int(card) - 1
        plane[card_index] = 1
    return plane

def encode_target(plane, targets):
    ''' Encode target and represerve it into plane

    Args:
        plane (array): 50 numpy array
        target(str): string of target card

    Returns:
        (array): 50 numpy array
    '''
    for index, target in enumerate(targets):
        target_index = int(target) - 1
        plane[index, target_index] = 1
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
        cards_list.append(card.rank)
    return cards_list

def targets2list(targets):
    ''' Get the corresponding string representation of cards

    Args:
        cards (list): list of UnoCards objects

    Returns:
        (string): string representation of cards
    '''
    targets_list = [targets['a1'].rank, targets['a2'].rank, targets['d1'].rank, targets['d2'].rank]

    return targets_list