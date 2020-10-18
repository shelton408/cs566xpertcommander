import State
import Player

HANDSIZES = {
    '1': 8,
    '2': 7,
    '3': 6  # 6 cards on hand for 3 or more players
}


class Game():
    '''
    Lets play The Game.
    '''

    def __init__(self, num_players):
        self.state = State()
        self.players = []
        for _ in range(num_players):
            handsize = HANDSIZES[str(3 if num_players > 2 else num_players)]
            hand = self.state.draw(handsize)
            self.players.append(Player(hand)))


    def play(self):

        turn_index = 0
        while self.state:
            current_player = self.players[turn_index]
            self.state = current_player.take_turn(self.state)
            turn_index += 1

        print(self.state)
