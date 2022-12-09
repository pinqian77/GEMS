import torch
import torch.nn.functional as F
from torch.optim import Adam

from network.rnd import RND
from network.oracle import Actor, Critic
from utils.util import soft_update, hard_update

class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, state_dim, action_dim, device):
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

        self.device = device

        # Define the actor
        self.actor = Actor(hidden_size, state_dim, action_dim).to(device)
        self.actor_target = Actor(hidden_size, state_dim, action_dim).to(device)

        # Define the critic
        self.critic = Critic(hidden_size, state_dim, action_dim).to(device)
        self.critic_target = Critic(hidden_size, state_dim, action_dim).to(device)

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

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def compute_rnd_reward(self, predict_feature, target_feature):
        rnd_reward = (target_feature - predict_feature).pow(2).sum(1) / 2
        return 1 + (rnd_reward - torch.mean(rnd_reward)) / torch.var(rnd_reward)

    def calc_action(self, state, action_noise=None):

        x = state.to(self.device)
        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu, sigma, action = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data
        sigma = sigma.data
        action = action.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(self.device)
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
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        ext_reward_batch = torch.cat(batch.ext_reward).to(self.device)
        int_reward_batch = torch.cat(batch.int_reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        # Get the actions and the state values to compute the targets
        _, __, next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        ext_reward_batch = ext_reward_batch.unsqueeze(1)
        int_reward_batch = int_reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)

        # do reward scaling
        int_reward_batch = 2 * (int_reward_batch - torch.min(int_reward_batch)) / (torch.max(int_reward_batch) - torch.min(int_reward_batch)) - 1

        # Compute total reward bacth
        reward_batch = ext_reward_batch + 0.1 * int_reward_batch
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()

        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        sampled_mu, sampled_sigma, sampled_action = self.actor(state_batch)

        # training oracle
        policy_loss = -self.critic(state_batch, sampled_action)
        policy_loss = policy_loss.mean()
        policy_loss.backward()

        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()