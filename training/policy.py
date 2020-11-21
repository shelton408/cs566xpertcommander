import torch
import torch.nn as nn
import torch.optim as optim 
# import torch.nn.functional as F
from torch.distributions.categorical import Categorical
# from utils import count_model_params

class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        self.num_actions = num_actions
        ############################## TODO: YOUR CODE BELOW ###############################
        ### 1. Build the Actor network as a torch.nn.Sequential module                   ###
        ###    with the following layers:                                                ###
        ###        (1) a Linear layer mapping from input dimension to hidden dimension   ###
        ###        (2) a Tanh non-linearity                                              ###
        ###        (3) a Linear layer mapping from hidden dimension to hidden dimension  ###
        ###        (4) a Tanh non-linearity                                              ###
        ###        (5) a Linear layer mapping from hidden dimension to number of actions ###
        ### HINT: We do not need an activation on the output, because the actor is       ###
        ###       predicting logits for categorical distribution.                        ###
        ####################################################################################
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
            #nn.Softmax(dim = -1)
        )
        ################################# END OF YOUR CODE #################################

    def forward(self, state):
        x = self.fc(state)
        return x


class Policy():
    '''
    Policy Class used for acting in the environment and updating the policy network.
    '''
    def __init__(self, num_inputs, num_actions, hidden_dim, learning_rate,
                 batch_size, policy_epochs, entropy_coef=0.001):
        self.actor = ActorNetwork(num_inputs, num_actions, hidden_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.policy_epochs = policy_epochs
        self.entropy_coef = entropy_coef

    def act(self, state, legal_action, Training=True):
        ############################## TODO: YOUR CODE BELOW ###############################
        ### 1. Run the actor network on the current state to get the action logits       ###
        ### 2. Build a Categorical(...) instance from the logits                         ###
        ### 3. Sample an action using the built-in sample() function of distribution.    ###
        ### Documentation of Categorical:                                                ###
        ### https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical
        ####################################################################################
        # all_prob = self.actor(state)
        # mask = torch.abs(torch.tensor(legal_action, dtype=torch.float32))
        # valid_prob = mask * all_prob
        #rescaled_valid_prob = valid_prob / torch.sum(valid_prob) #rescaled
        logits = self.actor(state)
        mask = torch.tensor(legal_action, dtype=torch.float32)
        mask[mask == 0] = -500
        mask[mask != -500] = 0
        legal_logits = torch.clamp(logits + mask, min=-50, max=50)

        dist = Categorical(logits = legal_logits)
        action = dist.sample() #sampling
        log_prob = dist.log_prob(action)
        ################################# END OF YOUR CODE #################################
        return action, log_prob

    def evaluate_actions(self, state, action, legal_actions):
        '''
        Evaluate the log probability of an action under the policy's output
        distribution for a given state.

        state -> tensor: [batch_size, obs_size]
        action -> tensor: [batch_size, 1]
        '''
        ############################## TODO: YOUR CODE BELOW ###############################
        ### This function is used for policy update to evaluate log_prob and entropy of  ###
        ### actor network.                                                               ###
        ### TODO: 
        ### 1. Compute logits and distribution for the given state (just like above).
        ### 2. Compute log probability of the given action under this distribution.
        ###    Hint: Input to the distribution should be in the shape [batch_size].
        ###          You may find `action.squeeze(...)` helpful.
        ### 3. Compute the entropy of the distribution.
        ####################################################################################
        #probs = self.actor(state) * torch.abs(torch.tensor(legal_actions, dtype=torch.float32))
        logits = self.actor(state)
        mask = torch.tensor(legal_actions, dtype=torch.float32)
        mask[mask == 0] = -500
        mask[mask != -500] = 0
        legal_logits = torch.clamp(logits + mask, min=-50, max=50)
        dist = Categorical(logits=legal_logits)
        log_prob = dist.log_prob(action.squeeze())
        entropy = dist.entropy()
        ################################# END OF YOUR CODE #################################
        return log_prob.view(-1, 1), entropy.view(-1, 1)

    def update(self, rollouts):
        '''
        Performing policy gradient update with maximum entropy regularization

        rollouts -> The storage buffer
        self.policy_epochs -> Number of times we train over the storage buffer
        '''
        for epoch in range(self.policy_epochs):
            data = rollouts.batch_sampler(self.batch_size)
            for sample in data:
                actions_batch, returns_batch, obs_batch, legal_actions_batch = sample
                # Compute Log probabilities and entropy for each sampled (state, action)
                log_probs_batch, entropy_batch = self.evaluate_actions(obs_batch, actions_batch, legal_actions_batch)

                ############################## TODO: YOUR CODE BELOW ###############################
                ### 4. Compute the mean loss for the policy update using action log-             ###
                ###     probabilities and policy returns                                         ###
                ### 5. Compute the mean entropy for the policy update                            ###
                ###    *HINT*: PyTorch optimizer is used to minimize by default.                 ###
                ###     The trick to maximize a quantity is to negate its corresponding loss.    ###
                ####################################################################################
                policy_loss = -torch.mean(torch.mul(log_probs_batch, returns_batch))
                entropy_loss = -torch.mean(entropy_batch)
                ################################# END OF YOUR CODE #################################

                loss = policy_loss + self.entropy_coef * entropy_loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.optimizer.step()

    @property
    def num_params(self):
        return sum(p.numel() for p in self.actor.parameters())
        # return count_model_params(self.actor)