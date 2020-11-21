import pfrl
import torch
import torch.nn
import gym
import the_game_gym
import numpy


# class QFunction(torch.nn.Module):
#
#     def __init__(self, obs_size, n_actions):
#         super().__init__()
#         self.l1 = torch.nn.Linear(obs_size, 50)
#         self.l2 = torch.nn.Linear(50, 50)
#         self.l3 = torch.nn.Linear(50, n_actions)
#
#     def forward(self, x):
#         h = x
#         h = torch.nn.functional.relu(self.l1(h))
#         h = torch.nn.functional.relu(self.l2(h))
#         h = self.l3(h)
#         return pfrl.action_value.DiscreteActionValue(h)


env = gym.make('1-player-static-full-game-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)

obs = env.reset()
print('initial observation:', obs)

action = env.action_space.sample()
obs, r, done, info = env.step(action)
print('next observation:', obs)
print('reward:', r)
print('done:', done)
print('info:', info)

obs_size = env.observation_space.n
n_actions = env.action_space.n
# q_func = QFunction(obs_size, n_actions)

q_func = torch.nn.Sequential(
    torch.nn.Linear(obs_size, 200),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 200),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 200),
    torch.nn.ReLU(),
    torch.nn.Linear(200, n_actions),
    pfrl.q_functions.DiscreteActionValueHead(),
)

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)


# Set the discount factor that discounts future rewards.
gamma = 0.9

# Use epsilon-greedy for exploration
explorer = pfrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)


# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 8)

gpu = -1


def phi(x):
    return x.astype(numpy.float32, copy=False)
# phi = lambda x: x.astype(numpy.float32, copy=False)


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

n_episodes = 10000
max_episode_len = 300
for i in range(1, n_episodes + 1):
    obs = env.reset()
    R = 0  # return (sum of rewards)
    t = 0  # time step
    D = 0
    Moves = 0
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        if reward > 0:
            Moves += 1
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break

    if i % 10 == 0:
        print('episode:', i, 'R:', R)
        print('drawpile:', len(env.game.state['drawpile']))
        print('num moves:', Moves, '\n')
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())
print('Finished.')

with agent.eval_mode():
    for i in range(10):
        obs = env.reset()
        R = 0
        t = 0
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)
            R += r
            t += 1
            reset = t == 200
            agent.observe(obs, r, done, reset)
            if done or reset:
                break
        print('evaluation episode:', i, 'R:', R)
        print('drawpile:', len(env.game.state['drawpile']))
