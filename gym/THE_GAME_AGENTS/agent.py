import gym
import the_game_gym
from gym import wrappers, logger
from pprint import pprint


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        import pdb; pdb.set_trace()
        
        return self.action_space.sample()


if __name__ == '__main__':

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('1-player-50c-game-v0')
    agent = RandomAgent(env.action_space)

    episode_count = 1
    total_reward = 0
    done = False

    rollout = 100
    ob = env.reset()

    for i in range(rollout):
        action = agent.act(ob)
        ob, reward, done, info = env.step(action)
        print('Reward: {}'.format(reward))
        total_reward += reward
        if done:
            break

    print(total_reward)
