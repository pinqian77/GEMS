import torch
import torch.nn.functional as F

from network.oracle import Actor
from network.student import HighLevelCritic, HighLevelPolicy, LowLevelPolicy, LowLevelCritic
from utils.util import bhattacharyya_gaussian_distance, soft_update, hard_update

class GEMS(object):
    def __init__(self, gamma, tau, ldf, hdf, hidden_size, state_dim, action_dim, device):

		# High level policy network that generate subgoal
        self.h_policy = HighLevelPolicy(hidden_size, state_dim, action_dim).to(device)
        self.h_policy_optimizer = torch.optim.Adam(self.h_policy.parameters(), lr=1e-3, weight_decay=1e-3)
        self.h_policy_target = HighLevelPolicy(hidden_size, state_dim, action_dim).to(device)
        self.h_policy_target.load_state_dict(self.h_policy.state_dict())
        hard_update(self.h_policy_target, self.h_policy)

        # Low level policy network
        self.l_policy = LowLevelPolicy(hidden_size, state_dim, action_dim).to(device)
        self.l_policy_optimizer = torch.optim.Adam(self.l_policy.parameters(), lr=1e-3, weight_decay=1e-3)
        self.l_policy_target = LowLevelPolicy(hidden_size, state_dim, action_dim).to(device)
        self.l_policy_target.load_state_dict(self.l_policy.state_dict())
        hard_update(self.l_policy_target, self.l_policy)

        # High level critic that score the subgoal's performance
        self.h_critic = HighLevelCritic(hidden_size, state_dim, action_dim).to(device)
        self.h_critic_optimizer = torch.optim.Adam(self.h_critic.parameters(), lr=1e-4, weight_decay=1e-4)
        self.h_critic_target = HighLevelCritic(hidden_size, state_dim, action_dim).to(device)
        self.h_critic_target.load_state_dict(self.h_critic.state_dict())
        hard_update(self.h_critic_target, self.h_critic)

		# Low level critic that score the action's performance
        self.l_critic = LowLevelCritic(hidden_size, state_dim, action_dim).to(device)
        self.l_critic_optimizer = torch.optim.Adam(self.l_critic.parameters(), lr=1e-4, weight_decay=1e-4)
        self.l_critic_target = LowLevelCritic(hidden_size, state_dim, action_dim).to(device)
        self.l_critic_target.load_state_dict(self.l_critic.state_dict())
        hard_update(self.l_critic_target, self.l_critic)

        # A trained network that generate goal
        self.oracle_actor = Actor(hidden_size, state_dim, action_dim).to(device)
        # self.oracle_actor.load_state_dict(torch.load('./oracle_mha_ngu_simhash_save.pkl'))
        self.oracle_actor.load_state_dict(torch.load('./oracle_cn.pkl'))

        self.tau = tau
        self.gamma = gamma
        self.low_distillation_factor = ldf
        self.high_distillation_factor = hdf
        self.device = device

    def get_goal(self, oracle_state):
        self.oracle_actor.eval()
        goal_mu, goal_sigma, goal = self.oracle_actor(oracle_state)

        return goal_mu.data, goal_sigma.data, goal.data

    def get_subgoal(self, normal_state):
        self.h_policy.eval()
        h_mu, h_sigma, subgoal = self.h_policy(normal_state)
        self.h_policy.train()
        h_mu = h_mu.data
        h_sigma = h_sigma.data
        subgoal = subgoal.data

        return h_mu, h_sigma, subgoal

    def get_action(self, normal_state, subgoal):
        self.l_policy.eval()
        l_mu, l_sigma, action = self.l_policy(normal_state, subgoal)
        self.l_policy.train()
        l_mu = l_mu.data
        l_sigma = l_sigma.data
        action = action.data

        return l_mu, l_sigma, action

    def update_low_params(self, batch):
        # Get tensors from the batch
        normal_state_batch = torch.cat(batch.normal_state).to(self.device)
        next_normal_state_batch = torch.cat(batch.next_normal_state).to(self.device)
        oracle_state_batch = torch.cat(batch.oracle_state).to(self.device)
        int_reward_batch = torch.cat(batch.int_reward).to(self.device)
        cosine_reward_batch = torch.cat(batch.cosine_reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        subgoal_batch = torch.cat(batch.subgoal).to(self.device)
        next_subgoal_batch = torch.cat(batch.next_subgoal).to(self.device)

        cosine_reward_batch = cosine_reward_batch.unsqueeze(1)
        int_reward_batch = int_reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)

        # do reward scaling
        int_reward_batch = 2 * (int_reward_batch - torch.min(int_reward_batch)) / (torch.max(int_reward_batch) - torch.min(int_reward_batch)) - 1

        # Compute total reward bacth
        reward_batch = cosine_reward_batch + 0.1 * int_reward_batch

		# Get the state and action value
        l_mu, l_sigma, next_action = self.l_policy_target(next_normal_state_batch, next_subgoal_batch)
        next_state_action_values = self.l_critic_target(next_normal_state_batch, next_subgoal_batch.detach(), next_action.detach())

		# Compute target value
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

		# Update the critic network                                           
        self.l_critic_optimizer.zero_grad()
        state_action_values = self.l_critic(normal_state_batch, subgoal_batch, action_batch)
        l_critic_loss = F.mse_loss(state_action_values, expected_values.detach())
        l_critic_loss.backward()
        self.l_critic_optimizer.step()

		# Update the actor network
        self.l_policy_optimizer.zero_grad()
        sampled_mu, sampled_sigma, sampled_action = self.l_policy(normal_state_batch, subgoal_batch)
        oralce_mu, oracle_sigma, oracle_action = self.get_goal(oracle_state_batch)

        # l_policy_loss = -self.l_critic(normal_state_batch, subgoal_batch, sampled_action) + self.low_distillation_factor * cosine_loss.to(device)
        distillation_loss = bhattacharyya_gaussian_distance(sampled_mu, sampled_sigma, oralce_mu, oracle_sigma)
        l_policy_loss = -self.l_critic(normal_state_batch, subgoal_batch, sampled_action) + self.low_distillation_factor * distillation_loss.to(self.device)
        
        l_policy_loss = l_policy_loss.mean()
        l_policy_loss.backward()
        self.l_policy_optimizer.step()

		# Soft update
        soft_update(self.l_critic_target, self.l_critic, self.tau)
        soft_update(self.l_policy_target, self.l_policy, self.tau)

        return l_critic_loss.item(), l_policy_loss.item(), distillation_loss.mean().item()

    def update_high_params(self, batch):
        # Get tensors from the batch
        normal_state_batch = torch.cat(batch.normal_state).to(self.device)
        oracle_state_batch = torch.cat(batch.oracle_state).to(self.device)
        next_normal_state_batch = torch.cat(batch.next_normal_state).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        subgoal_batch = torch.cat(batch.subgoal).to(self.device)

        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)

		# Get the next subgoal value
        h_mu, h_sigma, next_subgoal = self.h_policy_target(next_normal_state_batch)
        next_subgoal_values = self.h_critic_target(next_normal_state_batch, next_subgoal.detach())

		# Compute target subgoal value
        expected_subgoal_values = reward_batch + (1.0 - done_batch) * self.gamma * next_subgoal_values

        # Update the critic network
        self.h_critic_optimizer.zero_grad()
        subgoal_values = self.h_critic(normal_state_batch, subgoal_batch)
        h_critic_loss = F.mse_loss(subgoal_values, expected_subgoal_values.detach())
        h_critic_loss.backward()
        self.h_critic_optimizer.step()

		# Update the actor network
        self.h_policy_optimizer.zero_grad()
        sampled_mu, sampled_sigma, sampled_subgoal = self.h_policy(normal_state_batch)
        oralce_mu, oracle_sigma, oracle_action = self.get_goal(oracle_state_batch)

        distillation_loss = bhattacharyya_gaussian_distance(sampled_mu, sampled_sigma, oralce_mu, oracle_sigma)
        h_policy_loss = -self.h_critic(normal_state_batch, sampled_subgoal) + self.high_distillation_factor * distillation_loss.to(self.device)
        h_policy_loss = h_policy_loss.mean()
        h_policy_loss.backward()
        self.h_policy_optimizer.step()

        # Soft update
        soft_update(self.h_critic_target, self.h_critic, self.tau)
        soft_update(self.h_policy_target, self.h_policy, self.tau)

        return h_critic_loss.item(), h_policy_loss.item(), distillation_loss.mean().item()


	

