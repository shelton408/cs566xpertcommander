import torch
from training.instantiate import instantiate
from training.utils import ParamDict
from utils import plot_learning_curve, plot_testing
from cs566xpertcommander.the_game import Env
from training.duel_dqn import DuelDQN
from training.policy import Policy
from training.PPO import PPO

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
rollouts, ppo1 = instantiate(params)
ppo1.actor.load_state_dict(torch.load('./models/ppo_joint1.pt'))

rollouts, ppo2 = instantiate(params)
ppo2.actor.load_state_dict(torch.load('./models/ppo_joint2.pt'))

rollouts, ppo3 = instantiate(params)
ppo3.actor.load_state_dict(torch.load('./models/ppo_joint3.pt'))

rollouts, ppo4 = instantiate(params)
ppo4.actor.load_state_dict(torch.load('./models/ppo_joint4.pt'))

NUM_OF_PLAYERS = 4
config = {
    'num_players': NUM_OF_PLAYERS,
    'log_filename': './logs/policy_agent.log',
    'static_drawpile': False,
}
env = Env(config)

evaluations = []
num_iter = 50
for i in range(num_iter):  # lets play 50 games
    env.run_agents([ppo1, ppo2, ppo3, ppo4], False)
    # env.run_agents([policy])
    evaluations.append(env.get_num_cards_in_drawpile())
print('GAME OVER!')

plot_testing(evaluations, num_iter)