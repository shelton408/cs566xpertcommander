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

    def __init__(self, num_players, num_humans=0):
        self.state = State(num_players)
        self.players = []
        current_humans = 0
        for _ in range(num_players):
            handsize = HANDSIZES[str(3 if num_players > 2 else num_players)]
            hand = self.state.draw(handsize)
            new_player = Player(hand, current_humans < num_humans)
            current_humans += 1
            self.players.append(new_player)

    def play(self):

        # could this be for current_player in self.players?
        # or is turn_index necessary for some computation later?

        while self.state:
            for current_player in self.players:
                self.state = current_player.take_turn(self.state)

        print(self.state)


# for testing purposes, can remove whenever
if __name__ == '__main__':
    new_game = Game(num_players=1, num_humans=1)
    new_game.play()
