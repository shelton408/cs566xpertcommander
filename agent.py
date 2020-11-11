import torch
from training.policy import Policy
from training.instantiate import instantiate
import time
import sys
import logging
import numpy as np
from training.utils import AverageMeter, ParamDict

sys.path.append('../')
from cs556xpertcommander.the_game import Env


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

def encode(state):
    return list(state['decks'])

def train(env, rollouts, policy, params):
    rollout_time, update_time = AverageMeter(), AverageMeter()  # Loggers
    rewards, success_rate = [], []

    print("Training model with {} parameters...".format(policy.num_params))

    '''
    Training Loop
    '''
    for j in range(params.num_updates):
        ## Initialization
        avg_eps_reward, avg_success_rate = AverageMeter(), AverageMeter()
        done = False
        state, next_player = env.init_game()
        prev_obs = encode(state)
        prev_obs = torch.tensor(prev_obs, dtype=torch.float32)
        eps_reward = 0.
        start_time = time.time()

        ## Collect rollouts
        for step in range(rollouts.rollout_size):
            if done:
                # Store episode statistics
                avg_eps_reward.update(eps_reward)
                # if 'success' in info: 
                #     avg_success_rate.update(int(info['success']))

                # Reset Environment
                state, next_player = env.init_game()
                obs = torch.tensor(encode(state), dtype=torch.float32)
                eps_reward = 0.
            else:
                obs = prev_obs

            ############################## TODO: YOUR CODE BELOW ###############################
            ### 1. Call the policy to get the action for the current observation,            ###
            ### 2. Take one step in the environment (using the policy's action)              ###
            ####################################################################################
            prev_eval = env.game.num_playable()
            action, log_prob = policy.act(obs)
            if action <= len(env.game.state['legal_actions'][0]) - 1:
                state, next_player = env.step(action)
            else:
                state, next_player = env.step(np.random.randint(0, len(env.game.state['legal_actions'][0])))
            obs = encode(state)
            curr_eval = env.game.num_playable()
            reward = (prev_eval - curr_eval) / 7
            done = env.game.is_over()
            ################################# END OF YOUR CODE #################################


            ############################## TODO: YOUR CODE BELOW ###############################
            ### 3. Insert the sample <done, action, log_prob, reward, prev_obs> in the       ###
            ###    rollout storage. (requires just 1 line)                                   ###
            ### HINT:                                                                        ###
            ### - 'done' and 'reward' need to be converted to float32 tensors first          ###
            ### - Remember we are storing the previous observation because                   ###
            ###   that's what decided the policy's action                                    ###
            ####################################################################################
            done = torch.tensor(done, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            rollouts.insert(step, done, action, log_prob, reward, prev_obs)
            ################################# END OF YOUR CODE #################################

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

        ############################## TODO: YOUR CODE BELOW ###############################
        ### 5. Call the policy's update function using the collected rollouts            ###
        ####################################################################################
        policy.update(rollouts)
        ################################# END OF YOUR CODE #################################

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
        # if j % params.plotting_iters == 0 and j != 0:
        #     plot_learning_curve(rewards, success_rate, params.num_updates)
        #     log_policy_rollout(policy, params.env_name, pytorch_policy=True)
    # clear_output()   # this removes all training outputs to keep the notebook clean, DON'T REMOVE THIS LINE!
    return rewards, success_rate


NUM_OF_PLAYERS = 1

config = {
    'num_players': NUM_OF_PLAYERS,
    'log_filename': './logs/policy_agent.log'
}
logging.basicConfig(filename=config['log_filename'], filemode='w', level=logging.INFO)
env = Env(config)

rollouts, policy = instantiate(params)
rewards, success_rate = train(env, rollouts, policy, params)
print("Training completed!")
