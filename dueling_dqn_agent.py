from training.duel_dqn import DuelDQN
from training.instantiate import instantiate
import sys
import logging
from training.utils import ParamDict
from training.training import Trainer
from utils import plot_learning_curve, plot_testing
import warnings
import torch
warnings.filterwarnings("ignore")

sys.path.append('../')
from cs566xpertcommander.the_game import Env


# hyperparameters
policy_params = ParamDict(
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
params = ParamDict(
    policy_params=policy_params,
    rollout_size=5000,     # number of collected rollout steps per policy update
    num_updates=100,       # number of training policy iterations
    discount=0.999,        # discount factor
    plotting_iters=50,    # interval for logging graphs and policy rollouts
    # env_name=Env(),  # we are using a tiny environment here for testing
)

NUM_OF_PLAYERS = 1

config = {
    'num_players': NUM_OF_PLAYERS,
    'log_filename': './logs/policy_agent.log',
    'static_drawpile': False,
}
logging.basicConfig(filename=config['log_filename'], filemode='w', level=logging.INFO)
env = Env(config)

rollouts, dueling_dqn = instantiate(params)
trainer = Trainer()
rewards, deck_ends = trainer.train(env, rollouts, dueling_dqn, params)
print("Training completed!")

torch.save(dueling_dqn.Q.state_dict(), './models/duelingDQN.pt')

evaluations = []
num_iter = 50
for i in range(num_iter):  # lets play 50 games
    env.run_PG(dueling_dqn)
    evaluations.append(env.get_num_cards_in_drawpile())
print('GAME OVER!')
plot_learning_curve(deck_ends, params.num_updates)
plot_testing(evaluations, num_iter)
