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
        for _ in range(num_players):
            handsize = HANDSIZES[str(3 if num_players > 2 else num_players)]
            hand = self.state.draw(handsize)
            self.players.append(Player(hand))

    def play(self):

        # could this be for current_player in self.players?
        # or is turn_index necessary for some computation later?

        while self.state:
            for current_player in self.players:
                self.state = current_player.take_turn(self.state)

        print(self.state)


# for testing purposes, can remove whenever
if __name__ == '__main__':
    new_game = Game(1)
    new_game.play()
