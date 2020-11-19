import torch
import os

import rlcard
from rlcard.agents import DQNAgentPytorch
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

# Make environment
env = rlcard.make('thegame', config={'seed': 0})
eval_env = rlcard.make('thegame', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate/save plot
evaluate_every = 100
evaluate_num = 100
episode_num = 50000

# The intial memory size
memory_size = 50000
memory_init_size = 100

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/thegame_dqn_result/'

# Set a global seed
set_global_seed(0)

# Set up the agents
agent = DQNAgentPytorch(scope='dqn',
                        action_num=env.action_num,
                        replay_memory_size=memory_size,
                        replay_memory_init_size=memory_init_size,
                        train_every=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[512, 512])
env.set_agents([agent])
eval_env.set_agents([agent])

# Init a Logger to plot the learning curve
logger = Logger(log_dir)

for episode in range(episode_num):
    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    for ts in trajectories[0]:
        agent.feed(ts)

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('DQN')

# Save model
save_dir = 'models/thegame_dqn'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# torch.save(DQNAgentPytorch, os.path.join(save_dir, 'model'))