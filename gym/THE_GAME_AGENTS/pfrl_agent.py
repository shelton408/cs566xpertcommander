import logging
import sys
import pfrl
import torch
import torch.nn
import gym
import the_game_gym
import numpy
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')


env = gym.make('1-player-static-full-game-v0')
obs_size = env.observation_space.n
n_actions = env.action_space.n

q_func = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 200),
    torch.nn.Tanh(),
    torch.nn.Linear(200, 200),
    torch.nn.Tanh(),
    torch.nn.Linear(200, 200),
    torch.nn.Tanh(),
    torch.nn.Linear(200, n_actions),
    pfrl.q_functions.DiscreteActionValueHead(),
)

optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
gamma = 0.9

replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

gpu = -1

explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.1, random_action_func=env.action_space.sample)


q_func = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 200),
    torch.nn.Tanh(),
    torch.nn.Linear(200, 200),
    torch.nn.Tanh(),
    torch.nn.Linear(200, 200),
    torch.nn.Tanh(),
    torch.nn.Linear(200, n_actions),
    pfrl.q_functions.DiscreteActionValueHead(),
)


def phi(x):
    return x.astype(numpy.float32, copy=False)


agent = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    phi=phi,
    gpu=gpu,
)


pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=2000,           # Train the agent for 2000 steps
    eval_n_steps=None,       # We evaluate for episodes, not time
    eval_n_episodes=10,       # 10 episodes are sampled for each evaluation
    train_max_episode_len=200,  # Maximum length of each episode
    eval_interval=1000,   # Evaluate the agent after every 1000 steps
    outdir='result',      # Save everything to 'result' directory
)
