import torch
from training.instantiate import instantiate
from training.PPO import PPO
from training.training import Trainer
from training.utils import ParamDict
from utils import plot_learning_curve
from matplotlib import pyplot as plt
import numpy as np
from cs566xpertcommander.the_game import Env
from training.policy import Policy

import warnings
warnings.filterwarnings("ignore")

# hyperparameters
policy_params = ParamDict(
    policy_class=PPO,   # Policy class to use (replaced later)
    hidden_dim=128,         # dimension of the hidden state in actor network
    learning_rate=1e-3,    # learning rate of policy update
    batch_size=1024,       # batch size for policy update
    policy_epochs=25,       # number of epochs per policy update
    entropy_coef=0.002    # hyperparameter to vary the contribution of entropy loss
)
params = ParamDict(
    policy_params=policy_params,
    rollout_size=5000,     # number of collected rollout steps per policy update
    num_updates=200,       # number of training policy iterations
    discount=0.99,        # discount factor
    plotting_iters=100,    # interval for logging graphs and policy rollouts
    # env_name=Env(),  # we are using a tiny environment here for testing
)
rollouts, policy = instantiate(params)
# policy.actor.load_state_dict(torch.load('./models/policy.pt'))

NUM_OF_PLAYERS = 2
config = {
    'num_players': NUM_OF_PLAYERS,
    'log_filename': './logs/policy_agent.log',
    'static_drawpile': False,
}
env = Env(config)
agents = [policy, policy]
env.set_agents(agents)
trainer = Trainer()
useHints=True
rewards, deck_ends = trainer.train(env, rollouts, policy, params, use_hints=useHints)
print("Training completed!")

# torch.save(policy.actor.state_dict(), './models/policy.pt')
# policy.actor.load_state_dict(torch.load('./models/policy.pt'))

evaluations = []
num_iter = 50
for i in range(num_iter):  # lets play 50 games
    env.run_agents(agents, useHints)
    evaluations.append(env.get_num_cards_in_drawpile())
print('GAME OVER!')

fig, ax = plt.subplots(figsize=(10, 7)) 
bins = np.linspace(0, 100, 11)
ax.hist(evaluations, bins=bins) 

plt.show()

plot_learning_curve(evaluations, num_iter)