from training.policy import Policy
import torch
import torch.nn as nn
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super().__init__()

        ############################## TODO: YOUR CODE BELOW ###############################
        ### 1. Build the Ac network as a torch.nn.Sequential module                   ###
        ###    with the following layers:                                                ###
        ###        (1) a Linear layer mapping from input dimension to hidden dimension   ###
        ###        (2) a Tanh non-linearity                                              ###
        ###        (3) a Linear layer mapping from hidden dimension to hidden dimension  ###
        ###        (4) a Tanh non-linearity                                              ###
        ###        (5) a Linear layer mapping from hidden dimension to 1                 ###
        ### HINT: We do not need an activation on the output, because the actor is       ###
        ###       predicting a value, which can be any real number                       ###
        ####################################################################################
        self.fc = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        ################################# END OF YOUR CODE #################################

    def forward(self, state):
        x = self.fc(state)
        return x


class ACPolicy(Policy):
    def __init__(self, num_inputs, num_actions, hidden_dim, learning_rate, batch_size, policy_epochs,
                 entropy_coef=0.001, critic_coef=0.5):
        super().__init__(num_inputs, num_actions, hidden_dim, learning_rate, batch_size, policy_epochs, entropy_coef)

        self.critic = CriticNetwork(num_inputs, hidden_dim)
        
        ############################## TODO: YOUR CODE BELOW ###############################
        ### Create a common optimizer for actor and critic with the given learning rate  ###
        ### (requires 1-line of code)                                                    ###
        ####################################################################################
        self.optimizer = optim.Adam([{'params': self.critic.parameters()},
                                    {'params': self.actor.parameters()}], lr=learning_rate)
        ################################# END OF YOUR CODE #################################

        self.critic_coef = critic_coef
        
    def update(self, rollouts): 
        for epoch in range(self.policy_epochs):
            data = rollouts.batch_sampler(self.batch_size)
            
            for sample in data:
                actions_batch, returns_batch, obs_batch, legal_actions_batch = sample
                log_probs_batch, entropy_batch = self.evaluate_actions(obs_batch, actions_batch, legal_actions_batch)

                value_batch = self.critic(obs_batch)
                advantage = returns_batch - value_batch.detach()

                
                ############################## TODO: YOUR CODE BELOW ###############################
                ### 1. Compute the mean loss for the policy update using action log-             ###
                ###     probabilities and advantages.                                            ###
                ### 2. Compute the mean entropy for the policy update                            ###
                ### 3. Compute the critic loss as MSE loss between estimated value and expected  ###
                ###     returns.                                                                 ###
                ###    *HINT*: Carefully select the signs of each of the losses .                ###
                ####################################################################################
                policy_loss = -torch.mean(torch.mul(log_probs_batch, advantage))
                entropy_loss = -torch.mean(entropy_batch)
                critic_loss = torch.mean((returns_batch - value_batch) ** 2)
                ################################# END OF YOUR CODE #################################
                
                loss = policy_loss + \
                        self.critic_coef * critic_loss + \
                        self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    @property
    def num_params(self):
        return super().num_params + sum(p.numel() for p in self.critic.parameters())