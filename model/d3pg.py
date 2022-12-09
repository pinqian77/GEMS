import torch
from torch import nn
from torch.optim import Adam

import numpy as np

from network.rnd import RND
from network.student import D3PGActor, D3PGCritic
from utils.util import soft_update, hard_update

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _l2_project(next_distr_v, rewards_v, dones_mask_t, gamma, delta_z, n_atoms, v_min, v_max):
        next_distr = next_distr_v.data.cpu().numpy()
        rewards = rewards_v.data.cpu().numpy()
        dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)

        for atom in range(n_atoms):
            tz_j = np.minimum(v_max, np.maximum(v_min, rewards + (v_min + atom * delta_z) * gamma))
            b_j = (tz_j - v_min) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_mask = eq_mask.flatten().astype(np.int64)

            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            ne_mask = ne_mask.flatten().astype(np.int64)
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():
            proj_distr[dones_mask] = 0.0
            tz_j = np.minimum(v_max, np.maximum(v_min, rewards[dones_mask]))
            b_j = (tz_j - v_min) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_mask = eq_mask.flatten().astype(np.int64)
            eq_dones = dones_mask.copy()
            eq_dones[dones_mask] = eq_mask
            if eq_dones.any():
                proj_distr[eq_dones, l[eq_mask]] = 1.0
            ne_mask = u != l
            ne_mask = ne_mask.flatten().astype(np.int64)
            ne_dones = dones_mask.copy()
            ne_dones[dones_mask] = ne_mask
            if ne_dones.any():
                proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return proj_distr

class D3PG(object):

    def __init__(self, gamma, tau, hidden_size, state_dim, action_dim):
        """
        Arguments:
            gamma:          Discount factor
            tau:            Update factor for the actor and the critic
            hidden_size:    Number of units in the hidden layers of the actor and critic. Must be of length 2.
            num_inputs:     Size of the input states
            action_space:   The action space of the used environment. Used to clip the actions and 
                            to distinguish the number of outputs
            checkpoint_dir: Path as String to the directory to save the networks. 
                            If None then "./saved_models/" will be used
        """
        self.gamma = gamma
        self.tau = tau

        # Define the actor
        self.actor = D3PGActor(hidden_size, state_dim, action_dim).to(device)
        self.actor_target = D3PGActor(hidden_size, state_dim, action_dim).to(device)

        # Define the critic
        self.critic = D3PGCritic(hidden_size, state_dim, action_dim).to(device)
        self.critic_target = D3PGCritic(hidden_size, state_dim, action_dim).to(device)

        # Define the RND module
        self.rnd = RND(state_dim=state_dim, action_dim=action_dim).to(device)
        self.rnd_optimizer = Adam(self.rnd.parameters(), lr=1e-3)

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=1e-3,
                                    weight_decay=1e-3)  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=1e-4,
                                     weight_decay=1e-4
                                     )  # optimizer for the critic network

        self.value_criterion = nn.BCELoss(reduction='none')

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def compute_rnd_reward(self, predict_feature, target_feature):
        rnd_reward = (target_feature - predict_feature).pow(2).sum(1) / 2
        return 1 + (rnd_reward - torch.mean(rnd_reward)) / torch.var(rnd_reward)

    def calc_action(self, state, action_noise=None):

        x = state.to(device)
        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu, sigma, action = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data
        sigma = sigma.data
        action = action.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(device)
            action += noise

        return mu, sigma, action

    def update_params(self, batch):
        """
        Updates the parameters/networks of the agent according to the given batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update

        Arguments:
            batch:  Batch to perform the training of the parameters
        """
        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        ext_reward_batch = torch.cat(batch.ext_reward).to(device)
        int_reward_batch = torch.cat(batch.int_reward).to(device)
        done_batch = torch.cat(batch.done).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)

        # Compute the target
        ext_reward_batch = ext_reward_batch.unsqueeze(1)
        int_reward_batch = int_reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)

        # do reward scaling
        int_reward_batch = 2 * (int_reward_batch - torch.min(int_reward_batch)) / (torch.max(int_reward_batch) - torch.min(int_reward_batch)) - 1

        # Compute total reward bacth
        reward_batch = ext_reward_batch + 0.1 * int_reward_batch

        # Get the actions and the state values to compute the targets
        _, __, next_action_batch = self.actor_target(next_state_batch)

        # Predict Z distribution with target value network
        next_state_action_values_probs_batch = self.critic_target.get_probs(next_state_batch, next_action_batch.detach())
        target_z_projected_batch = _l2_project(next_distr_v=next_state_action_values_probs_batch,
                                            rewards_v=reward_batch,
                                            dones_mask_t=done_batch,
                                            gamma=self.gamma ** 5,
                                            n_atoms=23,
                                            v_min=-1,
                                            v_max=100,
                                            delta_z=100 / 22)
        target_z_projected_batch = torch.from_numpy(target_z_projected_batch).float().to(device)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic.get_probs(state_batch, action_batch)
        value_loss = self.value_criterion(state_action_batch, target_z_projected_batch)
        value_loss = value_loss.mean(axis=1)
        value_loss = value_loss.mean()
        # print("value_loss", value_loss)
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor networ
        self.actor_optimizer.zero_grad()
        sampled_mu, sampled_sigma, sampled_action = self.actor(state_batch)
        policy_loss = self.critic.get_probs(state_batch, sampled_action)
        policy_loss = policy_loss * torch.from_numpy(self.critic.z_atoms).float().to(device)
        policy_loss = -policy_loss.mean()
        # print("policy_loss", policy_loss)
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()