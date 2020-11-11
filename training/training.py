import sys
import torch
import torch.nn as nn
import logging
from collections.abc import Iterable
import time
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

sys.path.append('../')
from cs556xpertcommander.the_game import Env

#move to a utils class?
def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count         


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
    def __init__(self, num_players, agents, params, logfile='./training_log', obs_size=1):#replace obs_size with state size
        self.config = {
            'num_players': num_players,
            'log_filename': logfile
            #env_name for different envs later?
        }
        logging.basicConfig(filename=logfile,
                            filemode='w', level=logging.INFO)
        self.env = Env(self.config)
        self.env.set_agents(agents)
        self.params = params
        self.agents = agents
        self.obs_size = obs_size
        self.rollouts = RolloutStorage(self.params['rollout_size'], obs_size)

    def set_agents(self, agents):
        self.agents = agents

    def parse_state(self, state):
        out = []
        for k in state:
            if not k == 'legal_actions':
                out.append(state[k])
        ret = list(flatten(out))
        #this last part is not needed once we have a static state size
        while(len(ret) < self.obs_size):
            ret.append(0)
        ret = ret[:self.obs_size]
        return ret


    def reset_game(self):
        self.env = Env(self.config)
        self.env.set_agents(self.agents)
        self.env.init_game()
        return self.env.game.state, self.parse_state(self.env.game.state) #return state of first player as obs, if we allow agents to pick order somehow, this has to change

    def train(self):
        rollout_time, update_time = AverageMeter(), AverageMeter()  # Loggers
        rewards, success_rate = [], []

        for j in range(self.params['num_updates']):
            ## Initialization
            avg_eps_reward, avg_success_rate = AverageMeter(), AverageMeter()
            #minigrid resets the game after each rollout, we should either make rollout size big enough to reach endgame, or not
            done = False
            state, prev_obs = self.reset_game()
            prev_obs = torch.tensor(prev_obs, dtype=torch.float32)
            prev_eval = self.env.game.num_playable() #used to calculate reward
            eps_reward = 0.
            start_time = time.time()
            deck_end_sizes = []
            ## Collect rollouts
            for step in range(self.rollouts.rollout_size):
                if done:
                    # # Store episode statistics
                    avg_eps_reward.update(eps_reward)
                    deck_end_sizes.append(len(self.env.game.state['drawpile']))
                    # if 'success' in info: 
                    #     avg_success_rate.update(int(info['success']))

                    # Reset Environment
                    state, obs = self.reset_game()
                    obs = torch.tensor(obs, dtype=torch.float32)
                    prev_eval = self.env.game.num_playable() #used to calculate reward
                    eps_reward = 0.
                else:
                    obs = prev_obs

                #agent action
                #action, log_prob = agents[state['current_player']].act(state)
                action = self.agents[state['current_player']].step(state)
                log_prob = torch.tensor((.2),dtype=torch.float32)
                #obs, reward, done, info = self.env.step(action), info is useless to us since there is no success
                state, next_player = self.env.step(action)
                action_tensor = torch.tensor((action),dtype=torch.float32)
                obs = self.parse_state(state)

                #if our play reduces us by more than 5 playable cards, negatives reward, else positive
                eval = self.env.game.num_playable()
                reward = (eval - prev_eval)/5 + 1
                prev_eval = eval

                done = self.env.game.is_over()


                self.rollouts.insert(step, torch.tensor((done), dtype=torch.float32), action_tensor, log_prob, torch.tensor((reward), dtype=torch.float32), prev_obs)
                
                prev_obs = torch.tensor(obs, dtype=torch.float32)
                eps_reward += reward
            
            ############################## TODO: YOUR CODE BELOW ###############################
            ### 4. Use the rollout buffer's function to compute the returns for all          ###
            ###    stored rollout steps. Discount factor is given in 'params'                ###
            ### HINT: This requires just 1 line of code.                                     ###
            ####################################################################################
            self.rollouts.compute_returns(self.params['discount'])
            ################################# END OF YOUR CODE #################################
            
            rollout_done_time = time.time()

            # policy.update(rollouts) update here
            update_done_time = time.time()
            self.rollouts.reset()

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

