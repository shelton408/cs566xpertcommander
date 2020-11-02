from the_game import Game
import logging


class Env:
    '''
    The environment which agents play the game
    '''

    def __init__(self, config):
        self.game = Game(config['num_players'])
        self.agents = None
        logfile = './logs/run.log' if 'log_filename' not in config else config['log_filename']
        logging.basicConfig(filename=logfile,
                            filemode='w', level=logging.INFO)
        # logging.basicConfig(level=logging.DEBUG)
        # self.logger = logging.getLogger('Game')


    def set_agents(self, agents):
        self.agents = agents

    def _is_over(self):
        return self.game.is_over()

    def init_game(self):
        state, first_agent_id = self.game.init_game()
        return (state, first_agent_id)

    def step(self, action_id):
        return self.game.step(action_id)

    def eval(self):
        return {
            'Hand Eval': self.game.hand_eval(),
            'Drawpile Eval': self.game.drawpile_eval()
        }

    def run(self):
        state, agent_id = self.init_game()
        logging.info(' State for player {}: {}\n'.format(agent_id, str(state)))

        while not self._is_over():
            action_id = self.agents[agent_id].step(state)
            next_state, next_agent_id = self.step(action_id)
            state = next_state
            agent_id = next_agent_id
            logging.info(' State for player {}: {}\nEvaluation: {}\n'.format(agent_id, str(state), str(self.eval())))



    def _get_legal_actions(self):
        return self.game.get_legal_actions()

    def is_over(self):
        return self.game.is_over()

    def get_player_id(self):
        return self.game.get_player_id()

    def get_state(self, agent_id):
        return self.game.get_state(agent_id)
