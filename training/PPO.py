from training.policy import Policy
from training.AC_policy import ACPolicy
import torch
import torch.nn as nn
import torch.optim as optim


class PPO(ACPolicy):       
    def update(self, rollouts, use_hints=False): 
        self.clip_param = 0.2
        for epoch in range(self.policy_epochs):
            data = rollouts.batch_sampler(self.batch_size, get_old_log_probs=True)
            
            for sample in data:
                actions_batch, returns_batch, obs_batch, legal_actions_batch, old_log_probs_batch, hints = sample
                if use_hints:
                    log_probs_batch, entropy_batch = self.evaluate_actions(obs_batch, actions_batch, legal_actions_batch, hints)
                else:
                    log_probs_batch, entropy_batch = self.evaluate_actions(obs_batch, actions_batch, legal_actions_batch)
                
                value_batch = self.critic(obs_batch)
                
                advantage = returns_batch - value_batch.detach()
                old_log_probs_batch = old_log_probs_batch.detach()

                ############################## TODO: YOUR CODE BELOW ###############################
                ### Compute the following terms by following the equations given above           ###
                ### Useful functions: torch.exp(...), torch.clamp(...)
                ### Note: self.clip_param is the c in the above equations                        ###
                ### Compute the following terms by following the equations given above           ###
                ####################################################################################
                ratio = torch.exp(log_probs_batch - old_log_probs_batch)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                policy_loss = -torch.mean(torch.min(surr1, surr2))
                entropy_loss = -torch.mean(entropy_batch)
                critic_loss = torch.mean((returns_batch - value_batch) ** 2)
                ################################# END OF YOUR CODE #################################

                loss = policy_loss + \
                        self.critic_coef * critic_loss + \
                        self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward(retain_graph=False)
                self.optimizer.step()            