import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
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
                 epsilon_decay_steps=20000, gamma=0.999):
        self.Q = QNetwork(num_inputs, num_actions, hidden_dim)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.policy_epochs = policy_epochs
        self.entropy_coef = entropy_coef
        self.gamma = gamma

        self.num_actions = num_actions
        self.total_t = 0
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.epsilon_decay_steps = epsilon_decay_steps

    def act(self, state, legal_action, training=True):
        ############################## TODO: YOUR CODE BELOW ###############################
        ### 1. Run the actor network on the current state to get the action logits       ###
        ### 2. Build a Categorical(...) instance from the logits                         ###
        ### 3. Sample an action using the built-in sample() function of distribution.    ###
        ### Documentation of Categorical:                                                ###
        ### https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical
        ####################################################################################
        if training:
            qvals = self.Q(state)
            mask = torch.tensor(legal_action, dtype=torch.float32)

            epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
            prob = torch.ones(self.num_actions) * epsilon / self.num_actions
            q_values = qvals
            best_action = torch.argmax(q_values)
            prob[best_action] += (1.0 - epsilon)

            valid_prob = prob * mask

            # Categorical() will automatically scale the probs
            dist = Categorical(probs=valid_prob)

            action = dist.sample()  # sampling

            log_prob = dist.log_prob(action)
        else:
            with torch.no_grad:
                qvals = self.Q(state)
                mask = torch.tensor(legal_action, dtype=torch.float32)

                prob = torch.exp(qvals)
                valid_prob = prob * mask

                # Categorical() will automatically scale the probs
                dist = Categorical(probs=valid_prob)

                action = torch.argmax(valid_prob)

                log_prob = dist.log_prob(action)


        ################################# END OF YOUR CODE #################################
        return action, log_prob

    def evaluate_actions(self, state, action, legal_actions, next_state):
        '''
        Evaluate the log probability of an action under the policy's output
        distribution for a given state.

        state -> tensor: [batch_size, obs_size]
        action -> tensor: [batch_size, 1]
        '''

        q_batch = self.Q(state)
        action_q = q_batch.gather(1, action)

        next_q_batch = self.Q(next_state)
        next_q_batch_max = torch.max(next_q_batch, dim=1)[0]

        return action_q, next_q_batch_max

    def update(self, rollouts):
        '''
        Performing policy gradient update with maximum entropy regularization

        rollouts -> The storage buffer
        self.policy_epochs -> Number of times we train over the storage buffer
        '''
        for epoch in range(self.policy_epochs):
            data = rollouts.batch_sampler(self.batch_size, get_next_obs=True)
            for sample in data:
                actions_batch, returns_batch, obs_batch, legal_actions_batch, next_obs_batch = sample
                # Compute Log probabilities and entropy for each sampled (state, action)
                action_q, next_q_batch_max = self.evaluate_actions(obs_batch, actions_batch, legal_actions_batch, next_obs_batch)

                ############################## TODO: YOUR CODE BELOW ###############################
                ### 4. Compute the mean loss for the policy update using action log-             ###
                ###     probabilities and policy returns                                         ###
                ### 5. Compute the mean entropy for the policy update                            ###
                ###    *HINT*: PyTorch optimizer is used to minimize by default.                 ###
                ###     The trick to maximize a quantity is to negate its corresponding loss.    ###
                ####################################################################################
                expected_q = returns_batch + self.gamma * next_q_batch_max
                q_loss = F.mse_loss(action_q, expected_q)
                ################################# END OF YOUR CODE #################################

                loss = q_loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.optimizer.step()

    @property
    def num_params(self):
        return sum(p.numel() for p in self.Q.parameters())
        # return count_model_params(self.actor)