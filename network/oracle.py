import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from network.attention import MHAEncoder

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

def fan_in_uniform_init(tensor, fan_in=None):
    """Utility function for initializing actor and critic"""
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1. / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)


class Actor(nn.Module):
    def __init__(self, hidden_size, state_dim, action_dim):
        super(Actor, self).__init__()
        self.mha_encoder = MHAEncoder()

        # Layer 1
        self.linear1 = nn.Linear(state_dim, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Layer mu and sigma to get actions by reparameterization trick
        self.mu_layer = nn.Linear(hidden_size[1], action_dim)
        self.sigma_layer = nn.Linear(hidden_size[1], action_dim)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        fan_in_uniform_init(self.mu_layer.weight)
        fan_in_uniform_init(self.mu_layer.bias)

        nn.init.uniform_(self.sigma_layer.weight, 0.0, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.sigma_layer.bias, 0.0, BIAS_FINAL_INIT)
    
    def reparameterize(self, mu, sigma):
        """
        :param mu: mean from the encoder's latent space
        :param sigma: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * sigma) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        z = mu + (eps * std) # sampling as if coming from the input space
        return mu, std, z

    def forward(self, inputs, only_action=True):
        x = inputs

        # mha encode
        x, enc_self_attns = self.mha_encoder(x)
        x = x.reshape(x.shape[0], -1)

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # mu, sigma
        mu = self.mu_layer(x)
        sigma = self.sigma_layer(x)

        # Output
        _mu, _sigma, action = self.reparameterize(mu, sigma)
        # action = F.softmax(action)
        action = torch.tanh(action)
        
        return _mu, _sigma, action
        
class Critic(nn.Module):
    def __init__(self, hidden_size, state_dim, action_dim):
        super(Critic, self).__init__()
        self.mha_encoder = MHAEncoder()

        # Layer 1
        self.linear1 = nn.Linear(state_dim, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        # Layer 2
        # In the second layer the actions will be inserted also 
        self.linear2 = nn.Linear(hidden_size[0] + action_dim, hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], 1)

        # Weight Init
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)

        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)

        nn.init.uniform_(self.V.weight, -WEIGHTS_FINAL_INIT, WEIGHTS_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs

        # mha encode
        x, enc_self_attns = self.mha_encoder(x)
        x = x.reshape(x.shape[0], -1)

        # Layer 1
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, actions), 1)  # Insert the actions
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        # Output
        V = self.V(x)
        return V
