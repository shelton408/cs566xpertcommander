from the_game import Game
import numpy as np
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
        self.get_state()
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

    def get_state(self):
        '''
        Returns an encoded version of the state for the current player
        This is a N_Cards * 4 flat np.array
        (One hot encoded vector for what numbers are on hand) (N_Cards,)
        (One hot encoded vector for what numbers are on the Asc.DiscardDecks) (N_Cards + 1,) because starts at 1
        (One hot encoded vector for what numbers are on the Desc.DiscardDecks) (N_Cards + 1,) because starts at 100
        (One hot encoded vector for what numbers have been played already) (N_Cards)
        '''
        num_cards = self.game.state['number_of_cards']

        current_player_hand = self.game.state['hands'][self.game.state['current_player']] - 2  # because lowest card is 2
        hand = np.zeros(num_cards, dtype=int)
        hand[current_player_hand] = 1

        discard_decks = self.game.state['decks'] - 1

        asc_disc = np.zeros(num_cards + 2, dtype=int)
        asc_disc[discard_decks[:2]] = 1

        desc_disc = np.zeros(num_cards + 2, dtype=int)
        desc_disc[discard_decks[2:]] = 1

        played_cards = np.zeros(num_cards)
        played_cards[self.game.state['played_cards'] - 2] = 1
        encoded_state = np.concatenate((hand, asc_disc, desc_disc, played_cards), axis=None)

        return encoded_state
