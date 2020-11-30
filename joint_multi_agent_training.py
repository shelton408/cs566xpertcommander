import pickle

from training.PPO import PPO
from training.duel_dqn import DuelDQN
from training.policy import Policy
from training.instantiate import instantiate
import sys
import logging
from training.utils import ParamDict
from training.multi_agent_training import Trainer
from utils import plot_learning_curve, plot_testing
import warnings
import torch
warnings.filterwarnings("ignore")

sys.path.append('../')
from cs566xpertcommander.the_game import Env


# hyperparameters
policy_params = ParamDict(
    policy_class=Policy,   # Policy class to use (replaced later)
    hidden_dim=128,         # dimension of the hidden state in actor network
    learning_rate=1e-3,    # learning rate of policy update
    batch_size=1024,       # batch size for policy update
    policy_epochs=25,       # number of epochs per policy update
    entropy_coef=0.001    # hyperparameter to vary the contribution of entropy loss
)
policy_agent_params = ParamDict(
    policy_params=policy_params,
    rollout_size=5000,     # number of collected rollout steps per policy update
    num_updates=100,       # number of training policy iterations
    discount=0.999,        # discount factor
    plotting_iters=50,    # interval for logging graphs and policy rollouts
    # env_name=Env(),  # we are using a tiny environment here for testing
)

# hyperparameters
dueling_dqn_params = ParamDict(
    policy_class=DuelDQN,   # Policy class to use (replaced later)
    hidden_dim=128,         # dimension of the hidden state in actor network
    learning_rate=1e-3,    # learning rate of policy update
    batch_size=1024,       # batch size for policy update
    policy_epochs=25,       # number of epochs per policy update
    entropy_coef=0.001,    # hyperparameter to vary the contribution of entropy loss
    gamma=0.999,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps= int(0.8 * 5000 * 100)  # get to the final epsilon after 80% of training
)
dqn_agent_params = ParamDict(
    policy_params=dueling_dqn_params,
    rollout_size=5000,     # number of collected rollout steps per policy update
    num_updates=100,       # number of training policy iterations
    discount=0.999,        # discount factor
    plotting_iters=50,    # interval for logging graphs and policy rollouts
    # env_name=Env(),  # we are using a tiny environment here for testing
)

# hyperparameters
ppo_params = ParamDict(
    policy_class=PPO,   # Policy class to use (replaced later)
    hidden_dim=128,         # dimension of the hidden state in actor network
    learning_rate=1e-3,    # learning rate of policy update
    batch_size=1024,       # batch size for policy update
    policy_epochs=25,       # number of epochs per policy update
    entropy_coef=0.002    # hyperparameter to vary the contribution of entropy loss
)
ppo_agent_params = ParamDict(
    policy_params=ppo_params,
    rollout_size=5000,     # number of collected rollout steps per policy update
    num_updates=200,       # number of training policy iterations
    discount=0.99,        # discount factor
    plotting_iters=100,    # interval for logging graphs and policy rollouts
    # env_name=Env(),  # we are using a tiny environment here for testing
)

NUM_OF_PLAYERS = 4

config = {
    'num_players': NUM_OF_PLAYERS,
    'log_filename': './logs/policy_agent.log',
    'static_drawpile': False,
}
logging.basicConfig(filename=config['log_filename'], filemode='w', level=logging.INFO)
env = Env(config)


# policy_rollouts, policy = instantiate(policy_agent_params)
# dqn_rollouts, dqn = instantiate(dqn_agent_params)
# agents_list = [policy, dqn]
# rollouts_list = [policy_rollouts, dqn_rollouts]
# params_list = [policy_agent_params, dqn_agent_params]

ppo_rollouts1, ppo1 = instantiate(ppo_agent_params)
ppo_rollouts2, ppo2 = instantiate(ppo_agent_params)
ppo_rollouts3, ppo3 = instantiate(ppo_agent_params)
ppo_rollouts4, ppo4 = instantiate(ppo_agent_params)
agents_list = [ppo1, ppo2, ppo3, ppo4]
rollouts_list = [ppo_rollouts1, ppo_rollouts2, ppo_rollouts3, ppo_rollouts4]
params_list = [ppo_agent_params, ppo_agent_params, ppo_agent_params, ppo_agent_params]

env.set_agents(agents_list)
trainer = Trainer()
rewards, deck_ends = trainer.train(env, rollouts_list, agents_list, params_list)
print("Training completed!")

# torch.save(policy.actor.state_dict(), 'models/policy_joint.pt')
# torch.save(dqn.Q.state_dict(), 'models/dueling_dqn_joint.pt')
torch.save(ppo1.actor.state_dict(), 'models/policy_joint1.pt')
torch.save(ppo2.actor.state_dict(), 'models/policy_joint2.pt')
torch.save(ppo3.actor.state_dict(), 'models/policy_joint3.pt')
torch.save(ppo4.actor.state_dict(), 'models/policy_joint4.pt')

my_dict = {'joint_agent': deck_ends}
with open('pickle_files/ppo_agent_joint_train_4p.pickle', 'wb') as f:
    pickle.dump(my_dict, f)

plot_learning_curve(deck_ends, len(deck_ends))

# evaluations = []
# num_iter = 50
# for i in range(num_iter):  # lets play 50 games
#     env.run_PG(policy)
#     evaluations.append(env.get_num_cards_in_drawpile())
# print('GAME OVER!')
#
# plot_testing(evaluations, num_iter)
