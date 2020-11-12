from training.policy import Policy
from training.instantiate import instantiate
import sys
import logging
from training.utils import ParamDict
from training.training import Trainer

sys.path.append('../')
from cs566xpertcommander.the_game import Env


# hyperparameters
policy_params = ParamDict(
    policy_class=Policy,   # Policy class to use (replaced later)
    hidden_dim=32,         # dimension of the hidden state in actor network
    learning_rate=1e-3,    # learning rate of policy update
    batch_size=1024,       # batch size for policy update
    policy_epochs=4,       # number of epochs per policy update
    entropy_coef=0.001    # hyperparameter to vary the contribution of entropy loss
)
params = ParamDict(
    policy_params=policy_params,
    rollout_size=100,     # number of collected rollout steps per policy update
    num_updates=50,       # number of training policy iterations
    discount=0.99,        # discount factor
    plotting_iters=10,    # interval for logging graphs and policy rollouts
    # env_name=Env(),  # we are using a tiny environment here for testing
)

NUM_OF_PLAYERS = 1

config = {
    'num_players': NUM_OF_PLAYERS,
    'log_filename': './logs/policy_agent.log'
}
logging.basicConfig(filename=config['log_filename'], filemode='w', level=logging.INFO)
env = Env(config)

rollouts, policy = instantiate(params)
trainer = Trainer()
rewards, success_rate = trainer.train(env, rollouts, policy, params)
print("Training completed!")