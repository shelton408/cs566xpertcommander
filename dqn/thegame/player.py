
class TheGamePlayer(object):

    def __init__(self, player_id, np_random):
        ''' Initilize a player.

        Args:
            player_id (int): The id of the player
        '''
        self.np_random = np_random
        self.player_id = player_id
        self.hand = []

    def get_player_id(self):
        ''' Return the player's id
        '''

        return self.player_id