import gym
import the_game_gym
from pprint import pprint


class HumanAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        self._print_state(state, 0)
        action_id = int(input('>> Choose action index: '))
        return action_id

    def _print_state(self, state, player_id):
        '''Print state and available actions'''
        print('\n ====== Player num. {} ===='.format(player_id))
        print('\n======= Decks =========')
        print('0, asc: {}'.format(state['decks'][0]))
        print('1, asc: {}'.format(state['decks'][1]))
        print('2, dec: {}'.format(state['decks'][2]))
        print('3, dec: {}'.format(state['decks'][3]))

        print('\n====== Hand ========')
        print(state['hands'][state['current_player']])

        print('\n======= Actions available (card_id, deck_id) ======')

        legal_actions = [(a // 4, a % 4, v) for a, v in enumerate(state['legal_actions'][state['current_player']])]

        print(',\n'.join([str(idx) + ':\t' + str((action[0], action[1])) + ' legal: ' + str(bool(action[2])) for idx, action in enumerate(legal_actions)]))


if __name__ == '__main__':
    env = gym.make('1-player-50c-game-v0')
    agent = HumanAgent(env.action_space)
    done = False
    while not done:
        action = agent.act(env.game.state)
        ob, reward, done, info = env.step(action)
        print('Reward: {}'.format(reward))
        pprint(info)

    import pdb; pdb.set_trace()
