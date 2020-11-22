import gym
from gym import error, spaces, utils
from gym.utils import seeding
from the_game_gym import Game
import numpy as np


class GameEnv(gym.Env):
    def __init__(self, num_players, num_cards=98, is_static=False):
        self.game = Game(num_players, num_cards, is_static)
        self.action_space = spaces.Discrete((self.game.handsize * 4) + 1)
        self.observation_space = spaces.Box(0, 1, shape=(self.game.handsize + 4 + num_cards,))
        # self.max_avail_actions = self.game.handsize + 4 + num_cards
        # self.observation_space = spaces.Dict({
        # Box(0, 1, shape=(max_avail_actions, ))
        #     "action_mask": spaces.Discrete(self.game.handsize + 4 + num_cards),
        #     "avail_actions": spaces.Discrete(self.game.handsize + 4 + num_cards),
        # })

    def step(self, action) -> (object, float, bool, dict):
        state, score, extra_reward = self.game.step(action)
        reward = score if not extra_reward else score + self.game.eval()
        return self.get_encoded_state(state), reward, self.game.is_over(), state


    def reset(self):
        self.game.reset()
        return self.get_encoded_state(self.game.state)

    def render(self):
        return f'''
        ====== Player num. {self.game.state['current_player']} ====\n
        ======= Decks =========
        0, asc: {self.game.state['decks'][0]}
        1, asc: {self.game.state['decks'][1]}
        2, dec: {self.game.state['decks'][2]}
        3, dec: {self.game.state['decks'][3]}\n
        ====== Hand ========
        {self.game.state['hands'][self.game.state['current_player']]}
        '''

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
        # return spaces.Dict({
        #     "action_mask": state['legal_actions'][state['current_player']],
        #     "avail_actions": encoded_state
        # })

        return encoded_state
