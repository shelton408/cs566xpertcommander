import sys
import torch
import torch.nn as nn
import logging
import time
import numpy as np


from training.utils import AverageMeter, flatten
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions.categorical import Categorical

sys.path.append('../')
from cs566xpertcommander.the_game import Env

from training.duel_dqn import DuelDQN
from training.double_dqn import DoubleDQN

class RolloutStorage():
    def __init__(self, rollout_size, obs_size): #rollout size determines the number of turns taken per rollout
        self.rollout_size = rollout_size
        self.obs_size = obs_size
        self.action_size = 33
        self.hint_size = 33
        self.reset()

    def insert(self, step, done, action, log_prob, reward, obs, legal_actions, hint, next_obs=None):
        '''
        Inserting the transition at the current step number in the environment.
        '''
        self.done[step].copy_(done)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.rewards[step].copy_(reward)
        self.obs[step].copy_(obs)
        self.next_obs[step].copy_(next_obs)  # needed for DQN
        self.legal_actions[step].copy_(legal_actions)
        self.hints[step].copy_(hint)

    def reset(self):
        '''
        Initialize all storage buffers with zeros.
        '''
        self.done = torch.zeros(self.rollout_size, 1)
        self.returns = torch.zeros(self.rollout_size+1, 1, requires_grad=False)
        self.actions = torch.zeros(self.rollout_size, 1, dtype=torch.int64)  # Assuming Discrete Action Space
        self.log_probs = torch.zeros(self.rollout_size, 1)
        self.rewards = torch.zeros(self.rollout_size, 1)
        self.obs = torch.zeros(self.rollout_size, self.obs_size)
        self.next_obs = torch.zeros(self.rollout_size, self.obs_size)
        self.legal_actions = torch.zeros(self.rollout_size, self.action_size)
        self.hints = torch.zeros(self.rollout_size, self.hint_size)

    def compute_returns(self, gamma):
        '''
        Compute cumulative discounted returns from the current state to the end of the episode.
        '''
        self.last_done = (self.done == 1).nonzero().max()  # Find point of last episode's end in buffer.
        self.returns[self.last_done+1] = 0.  # Initialize the return at the end of last episode to be 0.

        # Accumulate discounted returns using dynamic programming.
        # Cumulative return = reward from current step + discounted future reward until end of episode.
        for step in reversed(range(self.last_done+1)):
            self.returns[step] = self.rewards[step] + \
                                self.returns[step + 1] * gamma * (1 - self.done[step])

    def batch_sampler(self, batch_size, get_old_log_probs=False, get_next_obs=False):
        '''
        Create a batch sampler of indices. Return actions, returns, observation for training.
        get_old_log_probs: This is required for PPO to recall the log_prob of the action w.r.t.
                           the policy that generated this transition.
        '''
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.last_done)),
            batch_size,
            drop_last=True)
        for indices in sampler:
            if get_old_log_probs:
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.legal_actions[indices], self.log_probs[indices]
            elif get_next_obs:
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.legal_actions[indices], self.next_obs[indices]
            else:
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.legal_actions[indices], self.hints[indices]


#params: rollout_size default this to 100 so we play an entire game?, num_updates
class Trainer():
    def __init__(self):#replace obs_size with state size
        self.obs_size = 4

    # def parse_state(self, state):
    #     out = []
    #     for k in state:
    #         if not k == 'legal_actions':
    #             out.append(state[k])
    #     ret = list(flatten(out))
    #     #this last part is not needed once we have a static state size
    #     while(len(ret) < self.obs_size):
    #         ret.append(0)
    #     ret = ret[:self.obs_size]
    #     return ret

    def reset_game(self, env):
        env.init_game()
        return env.game.state, env.get_encoded_state() #return state of first player as obs, if we allow agents to pick order somehow, this has to change

    def train(self, env, rollouts_list, agent_list, params_list, use_hints=False):
        rollout_time, update_time = AverageMeter(), AverageMeter()  # Loggers
        rewards, deck_avgs = [], []

        for i, agent in enumerate(agent_list):
            print("Training model {} with {} parameters...".format(i, agent.num_params))

        num_agent = len(agent_list)
        # assuming each rollouts storage is the same size
        rollout_size = rollouts_list[0].rollout_size
        # assuming each rollouts storage is the same size
        num_updates = params_list[0].num_updates

        '''
        Training Loop
        '''
        for j in range(num_updates):
            ## Initialization
            avg_eps_reward = AverageMeter()
            #minigrid resets the game after each rollout, we should either make rollout size big enough to reach endgame, or not
            done = False
            game_state, next_obs = self.reset_game(env)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            next_obs_list = [next_obs] * num_agent
            prev_eval = env.game.num_playable() #used to calculate reward
            eps_reward = 0.
            start_time = time.time()
            deck_end_sizes = []
            ## Collect rollouts
            for round_idx in range(rollout_size):  # assuming each rollouts storage is the same size
                if done:
                    # # Store episode statistics
                    avg_eps_reward.update(eps_reward)
                    deck_end_sizes.append(len(env.game.state['drawpile']))

                    # Reset Environment
                    game_state, obs = self.reset_game(env)
                    obs = torch.tensor(obs, dtype=torch.float32)
                    obs_list = [obs] * num_agent
                    prev_eval = env.game.num_playable() #used to calculate reward
                    eps_reward = 0.
                else:
                    obs_list = next_obs_list

                # default for when the game ends before the agent's turn
                action_list = [torch.tensor(32, dtype=torch.int32)] * num_agent
                log_prob_list = [torch.tensor(32, dtype=torch.float32)] * num_agent
                legal_actions_list = ([0] * 32 + [1]) * num_agent # can only skip turn
                hints_tensor_list = [torch.zeros(33, dtype=torch.float32)] * num_agent

                # for agent, rollouts, obs, next_obs, action, log_prob, legal_actions in \
                #         zip(agent_list, rollouts_list, obs_list, next_obs_list, action_list, log_prob_list, legal_actions_list):
                for a in range(num_agent):

                    #agent action
                    #action, log_prob = agents[state['current_player']].act(state)
                    curr_player = env.game.state['current_player']
                    original_legal_actions = env.game.state['legal_actions'][curr_player]
                    legal_actions_list[a] = original_legal_actions

                    obs_list[a] = env.get_encoded_state()
                    obs_list[a] = torch.tensor(obs_list[a], dtype=torch.float32)

                    if len(env.game.state['players']) <= 1:
                        hints_tensor_list[a] = torch.zeros(33, dtype=torch.float32)
                    else:
                        next_player = curr_player + 1
                        next_player = next_player if next_player < env.game.num_players else 0
                        hints_tensor_list[a] = torch.tensor(env.game.state['hints'][next_player], dtype=torch.float32)

                    if use_hints:
                        # 1-curr_player for 2 player game
                        action_list[a], log_prob_list[a] = agent_list[a].act(obs_list[a], legal_actions_list[a], hints_tensor_list[a])
                    else:
                        action_list[a], log_prob_list[a] = agent_list[a].act(obs_list[a], legal_actions_list[a])

                    if int(action_list[a]) <= len(legal_actions_list[a]) - 1:
                        state, next_player = env.step(action_list[a])
                    else:
                        raise ValueError('Action > length of legal actions')


                    # action = torch.argmax(rescaled_valid_prob)
                    # log_prob = torch.log(rescaled_valid_prob[action])
                    #obs, reward, done, info = self.env.step(action), info is useless to us since there is no success
                    # if action <= len(env.game.state['legal_actions'][0]) - 1:
                    #     game_state, next_player = env.step(action)
                    # else:
                    #     game_state, next_player = env.step(np.random.randint(0, len(env.game.state['legal_actions'][0])))
                    # action_tensor = torch.tensor(action, dtype=torch.float32)
                    next_obs_list[a] = env.get_encoded_state()
                    next_obs_list[a] = torch.tensor(next_obs_list[a], dtype=torch.float32)

                    if isinstance(agent_list[a], DuelDQN) or isinstance(agent_list[a], DoubleDQN):
                        agent_list[a].total_t += 1

                    done = env.game.is_over()
                    if done:
                        # update the next obs for all agents
                        for n in range(len(next_obs_list)):
                            next_obs_list[n] = next_obs
                        break

                # calculate reward after each round
                if True:
                    #if our play reduces us by more than 5 playable cards per player, negatives reward, else positive
                    curr_eval = env.game.num_playable()
                    reward = (curr_eval - prev_eval)/(5*num_agent) + 1
                    prev_eval = curr_eval
                    # if done:
                    #     reward = 50 / env.game.num_playable()
                    # else:
                    #     reward = 0
                else:
                    reward = -20

                eps_reward += reward
                # for agent, rollouts, obs, next_obs, action, log_prob, legal_actions in \
                #         zip(agent_list, rollouts_list, obs_list, next_obs_list, action_list, log_prob_list,
                #             legal_actions_list):
                for a in range(num_agent):
                    rollouts_list[a].insert(round_idx, torch.tensor((done), dtype=torch.float32), action_list[a], log_prob_list[a],
                                    torch.tensor((reward), dtype=torch.float32), obs_list[a],
                                    torch.tensor(legal_actions_list[a]),
                                    hints_tensor_list[a],
                                    torch.tensor(next_obs_list[a], dtype=torch.float32))



            for agent, rollouts, params in zip(agent_list, rollouts_list, params_list):
                rollouts.compute_returns(params.discount)
            rollout_done_time = time.time()

            for agent, rollouts, params in zip(agent_list, rollouts_list, params_list):
                if use_hints:
                    agent.update(rollouts, use_hints)
                else:
                    agent.update(rollouts)
                rollouts.reset()
            update_done_time = time.time()

            ## log metrics
            rewards.append(avg_eps_reward.avg)
            avg_deck_end_size = np.sum(deck_end_sizes) / len(deck_end_sizes)
            deck_avgs.append(avg_deck_end_size)
            rollout_time.update(rollout_done_time - start_time)
            update_time.update(update_done_time - rollout_done_time)
            print('it {}: avgR: {:.3f} -- rollout_time: {:.3f}sec -- update_time: {:.3f}sec'.format(j, avg_eps_reward.avg, 
                                                                                                    rollout_time.avg, 
                                                                                                    update_time.avg))
            # if j % self.params['plotting_iters'] == 0 and j != 0:
            #     plot_learning_curve(rewards, success_rate, params.num_updates)
            #     log_policy_rollout(policy, params['env_name'], pytorch_policy=True)
            print('av deck size: {:.3f}, games_played: {}'.format(avg_deck_end_size, len(deck_end_sizes)))
        return rewards, deck_avgs

