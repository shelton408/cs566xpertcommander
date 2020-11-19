import torch
import torch.nn as nn
import torch.optim as optim 
# import torch.nn.functional as F
from torch.distributions.categorical import Categorical
# from utils import count_model_params
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super().__init__()
        self.num_actions = num_actions

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(hidden_dim, num_actions),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, self.num_actions)
        )

    def forward(self, state):
        x = self.fc(state)
        values = self.value(x)
        advantages = self.advantage(x)
        qvals = values + (advantages - advantages.mean())
        return qvals

class DuelDQN():
    '''
    DuelDQN Class used for acting in the environment and updating the Q network.
    '''
    def __init__(self, num_inputs, num_actions, hidden_dim, learning_rate,
                 batch_size, policy_epochs, entropy_coef=0.001, epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,):
        self.Q = QNetwork(num_inputs, num_actions, hidden_dim)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.policy_epochs = policy_epochs
        self.entropy_coef = entropy_coef

        self.num_actions = num_actions
        self.total_t = 0
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon_decay_steps = epsilon_decay_steps

    def act(self, state, legal_action):
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
        qvals = self.Q(state)
        mask = torch.tensor(legal_action, dtype=torch.float32)

        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        A = np.ones(self.num_actions, dtype=float) * epsilon / self.num_actions
        q_values = qvals
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)

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
        logits = self.Q(state)
        mask = torch.tensor(legal_actions, dtype=torch.float32)
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
        return sum(p.numel() for p in self.Q.parameters())
        # return count_model_params(self.actor)