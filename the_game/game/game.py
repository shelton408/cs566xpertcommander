from the_game.game.state import State
from the_game.player.player import Player

HANDSIZES = {
    '1': 8,
    '2': 7,
    '3': 6  # 6 cards on hand for 3 or more players
}


class Game:
    """
    Lets play The Game.
    """

    def __init__(self, num_players):
        self.state = State(num_players)
        self.players = []
        for num in range(num_players):
            handsize = HANDSIZES[str(3 if num_players > 2 else num_players)]
            hand = self.state.draw(handsize)
            new_player = Player(num, hand)
            self.players.append(new_player)


    def __str__(self):
        players_str = ''
        for p in self.players:
            players_str += (str(p) + '\n')

        return '{}\n{}'.format(str(self.state), players_str)


    def play(self):
        '''
        The game loop itself. It loops through players until game is over.
        '''
        while not self.isGameOver():
            for current_player in self.players:
                if not self.isGameOver() and current_player.can_play():  # Player can have no cards on hand so it skips him
                    self.state = current_player.take_turn(self.state)

            print(self)
        print('\nGAME OVER\n')

    def isGameOver(self):
        '''
        Boolean function that checks if the game is over.
        Either by state being invalid (game lost) or all cards been played (game won)
        '''
        if self.state.isEndState\
                or (len(self.state.drawpile) == 0 and all([len(p.hand) == 0 for p in self.players])):
            print('GAME is OVER')
            print(self.state.isEndState)
            return True
        else:
            return False
