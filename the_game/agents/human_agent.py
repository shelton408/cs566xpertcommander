class HumanAgent:
    '''A human agent'''

    def __init__(self, id):
        self.id = id

    def step(self, state):
        '''Human Agent will display the state and make move'''
        _print_state(state, self.id)
        # import pdb; pdb.set_trace()

        action_id = int(input('>> Choose action index: '))
        # TODO: Validate input
        return action_id



def _print_state(state, player_id):
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
    print(','.join([str(idx) + ': ' + str(action) for idx, action in enumerate(state['legal_actions'][state['current_player']])]))
