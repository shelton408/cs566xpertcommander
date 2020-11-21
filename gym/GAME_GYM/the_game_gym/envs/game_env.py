import gym
from gym import error, spaces, utils
from gym.utils import seeding
from the_game_gym import Game
import numpy as np


class GameEnv(gym.Env):
    def __init__(self, num_players, num_cards=98, is_static=False):
        self.game = Game(num_players, num_cards, is_static)
        self.action_space = spaces.Discrete((self.game.handsize * 4) + 1)
        self.observation_space = spaces.Discrete(self.game.handsize + 4 + num_cards)

    def step(self, action) -> (object, float, bool, dict):
        state, score, extra_reward = self.game.step(action)
        reward = score if not extra_reward else score + self.game.eval()
        return self.get_encoded_state(state), reward, self.game.is_over(), state


    def reset(self):
        self.game.reset()
        return self.get_encoded_state(self.game.state)

    def render(self):
        pass

    def _get_state(self):
        return self.observation


    def get_encoded_state(self, state):
        num_cards = state['number_of_cards']

        # normalize hand and discard decks
        hand = state['hands'][state['current_player']] / 100
        discard_decks = (state['decks']) / 100

        unplayed_cards = np.ones(num_cards)
        unplayed_cards[state['played_cards'] - 2] = 0

        encoded_state = np.concatenate((hand, discard_decks, unplayed_cards), axis=None)
        return encoded_state
