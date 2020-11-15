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


class RolloutStorage():
    def __init__(self, rollout_size, obs_size): #rollout size determines the number of turns taken per rollout
        self.rollout_size = rollout_size
        self.obs_size = obs_size
        self.reset()

    def insert(self, step, done, action, log_prob, reward, obs):
        '''
        Inserting the transition at the current step number in the environment.
        '''
        self.done[step].copy_(done)
        self.actions[step].copy_(action)
        self.log_probs[step].copy_(log_prob)
        self.rewards[step].copy_(reward)
        self.obs[step].copy_(obs)

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

    def batch_sampler(self, batch_size, get_old_log_probs=False):
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
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.log_probs[indices]
            else:
                yield self.actions[indices], self.returns[indices], self.obs[indices]


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
        return env.game.state, env.get_state() #return state of first player as obs, if we allow agents to pick order somehow, this has to change

    def train(self, env, rollouts, policy, params):
        rollout_time, update_time = AverageMeter(), AverageMeter()  # Loggers
        rewards, success_rate = [], []

        print("Training model with {} parameters...".format(policy.num_params))

        '''
        Training Loop
        '''
        for j in range(params.num_updates):
            ## Initialization
            avg_eps_reward, avg_success_rate = AverageMeter(), AverageMeter()
            #minigrid resets the game after each rollout, we should either make rollout size big enough to reach endgame, or not
            done = False
            game_state, prev_obs = self.reset_game(env)
            prev_obs = torch.tensor(prev_obs, dtype=torch.float32)
            prev_eval = env.game.num_playable() #used to calculate reward
            eps_reward = 0.
            start_time = time.time()
            deck_end_sizes = []
            ## Collect rollouts
            for step in range(rollouts.rollout_size):
                if done:
                    # # Store episode statistics
                    avg_eps_reward.update(eps_reward)
                    deck_end_sizes.append(len(env.game.state['drawpile']))
                    # if 'success' in info: 
                    #     avg_success_rate.update(int(info['success']))

                    # Reset Environment
                    game_state, obs = self.reset_game(env)
                    obs = torch.tensor(obs, dtype=torch.float32)
                    prev_eval = env.game.num_playable() #used to calculate reward
                    eps_reward = 0.
                else:
                    obs = prev_obs

                #agent action
                #action, log_prob = agents[state['current_player']].act(state)

                action, log_prob = policy.act(obs, env.game.state['legal_actions'][0])
                
                if int(action) <= len(env.game.state['legal_actions'][0]) - 1:
                    state, next_player = env.step(action)
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
                obs = env.get_state()

                #if our play reduces us by more than 5 playable cards, negatives reward, else positive
                curr_eval = env.game.num_playable()
                reward = (curr_eval - prev_eval)/5 + 1
                prev_eval = curr_eval
                done = env.game.is_over()
                rollouts.insert(step, torch.tensor((done), dtype=torch.float32), action, log_prob, torch.tensor((reward), dtype=torch.float32), prev_obs)
                
                prev_obs = torch.tensor(obs, dtype=torch.float32)
                eps_reward += reward
            
            ############################## TODO: YOUR CODE BELOW ###############################
            ### 4. Use the rollout buffer's function to compute the returns for all          ###
            ###    stored rollout steps. Discount factor is given in 'params'                ###
            ### HINT: This requires just 1 line of code.                                     ###
            ####################################################################################
            rollouts.compute_returns(params.discount)
            ################################# END OF YOUR CODE #################################
            
            rollout_done_time = time.time()

            policy.update(rollouts)
            update_done_time = time.time()
            rollouts.reset()

            ## log metrics
            rewards.append(avg_eps_reward.avg)
            if avg_success_rate.count > 0:
                success_rate.append(avg_success_rate.avg)
            rollout_time.update(rollout_done_time - start_time)
            update_time.update(update_done_time - rollout_done_time)
            print('it {}: avgR: {:.3f} -- rollout_time: {:.3f}sec -- update_time: {:.3f}sec'.format(j, avg_eps_reward.avg, 
                                                                                                    rollout_time.avg, 
                                                                                                    update_time.avg))
            # if j % self.params['plotting_iters'] == 0 and j != 0:
            #     plot_learning_curve(rewards, success_rate, params.num_updates)
            #     log_policy_rollout(policy, params['env_name'], pytorch_policy=True)
            print('Deck end sizes:{}'.format(deck_end_sizes))
        return rewards, success_rate

