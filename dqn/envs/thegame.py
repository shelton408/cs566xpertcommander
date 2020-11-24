import numpy as np

from rlcard.envs import Env
from rlcard.games.thegame import Game
from rlcard.games.thegame.utils import encode_card, encode_target, cards2list, targets2list
from rlcard.games.thegame.utils import ACTION_SPACE, ACTION_LIST


class TheGameEnv(Env):

    def __init__(self, config):
        self.name = 'thegame'
        self.game = Game()
        super().__init__(config)
        self.state_shape = (5, config['deck_size']+2)
    
    'State encoding can impact the performance. This part can be modified'
    def _extract_state(self, state):
        obs = np.zeros(self.state_shape, dtype=int)
        encode_card(obs[0, :], state['hand'])
        encode_target(obs[1:5, :], state['target'])
        # encode_card(obs[-1, :], state['playable_cards'])
        legal_action_id = self._get_legal_actions()

        extracted_state = {'obs': obs, 'legal_actions': legal_action_id}
        if self.allow_raw_data:
            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [
                a for a in state['legal_actions']]
        if self.record_action:
            extracted_state['action_record'] = self.action_recorder
        return extracted_state

    def get_payoffs(self):

        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        legal_ids = self._get_legal_actions()
        if action_id in legal_ids:
            return ACTION_LIST[action_id]
        return ACTION_LIST[np.random.choice(legal_ids)]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = [ACTION_SPACE[action] for action in legal_actions]
        return legal_ids

    # this appears to never get called
    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['player_num'] = self.game.get_player_num()
        state['hand_cards'] = [cards2list(player.hand)
                               for player in self.game.players]
        state['played_cards'] = cards2list(self.game.round.played_cards)
        state['target'] = target2list(self.game.round.target)
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.round.get_legal_actions(
            self.game.players, state['current_player'])
        return state
