import numpy as np

from rlcard.envs import Env
from rlcard.games.thegame import Game
from rlcard.games.thegame.utils import encode_hand, encode_card, encode_target, cards2list, targets2list
from rlcard.games.thegame.utils import ACTION_SPACE, ACTION_LIST

HANDSIZES = {
    '1': 8,
    '2': 7,
    '3': 6  # 6 cards on hand for 3 or more players
}

class TheGameEnv(Env):

    def __init__(self, config):
        self.name = 'thegame'
        num_players = config['num_players']
        hand_size = HANDSIZES[str(3 if num_players > 2 else num_players)]
        self.game = Game(num_players=num_players,
                         deck_size=config['deck_size'],
                         hand_size=hand_size)
        super().__init__(config)

        self.state_shape = hand_size + 4 + config['deck_size']  # cards_in_hand + target + playable_card = 8 + 4 + 98 = 110
    
    'State encoding can impact the performance. This part can be modified'
    '''
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
    '''

    def _extract_state(self, state):
        obs = np.zeros(self.state_shape, dtype=float)
        encode_hand(obs[:8], state['hand'])
        encode_target(obs[8:12], state['target'])
        encode_card(obs[12:], state['playable_cards'])

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

    def get_playable_cards(self):

        return self.game.get_playable_cards()

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
        state['target'] = targets2list(self.game.round.target)
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.round.get_legal_actions(
            self.game.players, state['current_player'])
        return state
