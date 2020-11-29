import sys
import torch
import torch.nn as nn
import logging
import time
import numpy as np

from training.utils import AverageMeter, flatten
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions.categorical import Categorical
from utils import plot_learning_curve

sys.path.append('../')
from cs566xpertcommander.the_game import Env

from training.duel_dqn import DuelDQN

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
        self.legal_actions[step].copy_(legal_actions)
        self.hints[step].copy_(hint)
        self.next_obs[step].copy_(next_obs)  # needed for DQN

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
        self.legal_actions = torch.zeros(self.rollout_size, self.action_size)
        self.hints = torch.zeros(self.rollout_size, self.hint_size)
        self.next_obs = torch.zeros(self.rollout_size, self.obs_size)

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
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.legal_actions[indices], self.log_probs[indices], self.hints[indices]
            elif get_next_obs:
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.legal_actions[indices], self.next_obs[indices]
            else:
                yield self.actions[indices], self.returns[indices], self.obs[indices], self.legal_actions[indices], self.hints[indices]

#params: rollout_size default this to 100 so we play an entire game?, num_updates
class Trainer():

    def reset_game(self, env):
        env.init_game()
        return env.game.state, env.get_encoded_state() #return state of first player as obs, if we allow agents to pick order somehow, this has to change

    def train(self, env, rollouts, policy, params, use_hints=False):
        rollout_time, update_time = AverageMeter(), AverageMeter()  # Loggers
        rewards, deck_avgs, av_ds = [], [], []

        print("Training model with {} parameters...".format(policy.num_params))

        '''
        Training Loop
        '''
        train_start_time = time.time()

        for j in range(params.num_updates):
            ## Initialization
            avg_eps_reward = AverageMeter()
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

                    # Reset Environment
                    game_state, obs = self.reset_game(env)
                    obs = torch.tensor(obs, dtype=torch.float32)
                    prev_eval = env.game.num_playable() #used to calculate reward
                    eps_reward = 0.
                else:
                    obs = prev_obs

                #agent action
                #action, log_prob = agents[state['current_player']].act(state)
                curr_player = env.game.state['current_player']
                original_legal_actions = env.game.state['legal_actions'][curr_player]

                legal_actions = original_legal_actions

                if len(env.game.state['players']) <= 1:
                    hints_tensor = torch.zeros(33, dtype=torch.float32)
                else:
                    hints_tensor = torch.tensor(env.game.state['hints'][1 - curr_player], dtype=torch.float32)

                if use_hints:
                    action, log_prob = policy.act(obs, legal_actions, hints_tensor) # 1-curr_player for 2 player game
                else:
                    action, log_prob = policy.act(obs, legal_actions)
                
                if original_legal_actions[int(action)]:
                    state, next_player = env.step(action)
                    
                obs = env.get_encoded_state()

                if original_legal_actions[int(action)]:
                    #if our play reduces us by more than 5 playable cards, negatives reward, else positive
                    curr_eval = env.game.num_playable()
                    reward = (curr_eval - prev_eval)/5 + 1
                    prev_eval = curr_eval
                else:
                    reward = -20
                done = env.game.is_over()

                rollouts.insert(step, torch.tensor((done), dtype=torch.float32), action, log_prob,
                                torch.tensor((reward), dtype=torch.float32), prev_obs, torch.tensor(legal_actions),
                                hints_tensor, torch.tensor(obs, dtype=torch.float32))

                if isinstance(policy, DuelDQN):
                    policy.total_t += 1

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
            if use_hints:
                policy.update(rollouts, use_hints)
            else:
                policy.update(rollouts)
            update_done_time = time.time()
            rollouts.reset()

            ## log metrics
            rewards.append(avg_eps_reward.avg)
            avg_deck_end_size = np.sum(deck_end_sizes) / len(deck_end_sizes)
            deck_avgs.append(avg_deck_end_size)
            rollout_time.update(rollout_done_time - start_time)
            update_time.update(update_done_time - rollout_done_time)
            av_ds.append(av_deck_end_size)
            print('it {}: avgR: {:.3f} -- rollout_time: {:.3f}sec -- update_time: {:.3f}sec'.format(j, avg_eps_reward.avg, 
                                                                                                    rollout_time.avg, 
                                                                                                    update_time.avg))

            if (j + 1) % params.plotting_iters == 0 and j != 0:
                  plot_learning_curve(av_ds, j+1)
            # if j % self.params['plotting_iters'] == 0 and j != 0:
            #     plot_learning_curve(rewards, success_rate, params.num_updates)
            #     log_policy_rollout(policy, params['env_name'], pytorch_policy=True)
            print('av deck size: {}, games_played: {}'.format(avg_deck_end_size, len(deck_end_sizes)))
            
        return rewards, deck_avgs
