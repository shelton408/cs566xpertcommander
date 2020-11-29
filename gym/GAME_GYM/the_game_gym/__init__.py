# flake8: noqa

from .game import Game
from gym.envs.registration import register

register(
    id='1-player-full-game-v0',
    entry_point='the_game_gym.envs:GameEnv',
    kwargs={'num_players': 1}
)

register(
    id='1-player-static-full-game-v0',
    entry_point='the_game_gym.envs:GameEnv',
    kwargs={'num_players': 1, 'is_static': True}
)

register(
    id='2-player-full-game-v0',
    entry_point='the_game_gym.envs:GameEnv',
    kwargs={'num_players': 2}
)

register(
    id='2-player-static-full-game-v0',
    entry_point='the_game_gym.envs:GameEnv',
    kwargs={'num_players': 2, 'is_static': True}
)

register(
    id='1-player-50c-game-v0',
    entry_point='the_game_gym.envs:GameEnv',
    kwargs={'num_players': 1, 'num_cards': 50}
)
