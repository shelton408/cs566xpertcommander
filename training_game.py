#currently depreciated dont use
from training.training import Trainer
from the_game import RandomAgent
from the_game import Env
NUM_PLAYERS = 1
agents = []
for i in range(NUM_PLAYERS):
    agents.append(RandomAgent(i))

params = {
    'rollout_size': 500,
    'num_updates': 5,
    'discount': 0.99,
    'plotting_iters': 10,
    'env_name': '1p',
}
trainer = Trainer(NUM_PLAYERS, agents, params)
OBS_SIZE = len(trainer.parse_state(trainer.reset_game()[0]))
trainer = Trainer(NUM_PLAYERS, agents, params, obs_size=OBS_SIZE)
rewards, success_rate = trainer.train()
print(rewards)