import torch
from training.instantiate import instantiate
from training.PPO import PPO
from training.training import Trainer
from training.utils import ParamDict
from utils import plot_learning_curve
from cs566xpertcommander.the_game import Env
from training.policy import Policy
import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# hyperparameters
policy_params = ParamDict(
    policy_class=Policy,   # Policy class to use (replaced later)
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
policies = ['./models/policy.pt', './models/policy_m2.pt', './models/policy_m3.pt', './models/policy_m4.pt', './models/policy_m5.pt']
colors = ['red', 'blue', 'green', 'black', 'darkviolet']
cnt = 0

for pol in policies:

    print('Using {} policy'.format(pol))
    policy.actor.load_state_dict(torch.load(pol))

    num_iter = 50
    avg_drawpile_size = []

    for i in range(1, 6):

        print('Playing with {} number of players'.format(i))
        evaluations = []
        NUM_OF_PLAYERS = i
        config = {
            'num_players': NUM_OF_PLAYERS,
            'log_filename': './logs/policy_agent.log',
            'static_drawpile': False
        }
        env = Env(config)
        agents = [policy] * i
        for j in range(num_iter):  # lets play 50 games
            if pol == './models/policy.pt':
                useHints=False
            else:
                useHints=True
            env.run_agents(agents, use_hints=useHints)
            evaluations.append(env.get_num_cards_in_drawpile())
        
        avg_drawpile_size.append(np.mean(evaluations))

    plt.plot(list(range(1, 6)), avg_drawpile_size, label='Trained using {} players'.format(cnt + 1), color=colors[cnt])
    cnt+=1
    print()

plt.xlabel('Number of players while playing')
plt.ylabel('Average drawpile size')
plt.title('Performance of best agents during multi-player games')
plt.legend()
plt.show()