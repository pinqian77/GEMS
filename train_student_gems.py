import h5py
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from model.gems import GEMS
from env.portofolio import PortfolioEnvMain
from utils.replay_memory import ReplayMemory, TransitionL, TransitionH

def normalize_obs_diff(obs,scaling=1):
    """
    Inputs:
        obs:[asset_num,window_length,feature_num] obs must bigger than zero
    Return:
        normalized_obs:[asset_num,window_length,feature_num]
    """
    denominator = obs[:,-1:,:]
    out = (( obs[:,:,:] / (denominator+1e-8) )-1)*scaling

    return out

def cosine_distance(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    similiarity = np.dot(a, b.T) / (a_norm * b_norm) 
    dist = 1. - similiarity
    return dist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {}".format(device))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default = 1, type = int)

parser.add_argument("--env", default = "StockTradingEnv_v4")                        # includes [v1, v2, v3, v4]
parser.add_argument("--feature_num", default = 9, type = int)                       # number of features (default: 9)
parser.add_argument("--window_size", default = 20, type=int)                        # window size
parser.add_argument("--step_size", default = 1, type = int)                         # step size
parser.add_argument("--low_distillation", default = 0.5)                            # Low level distillation factor (default: 0.5)
parser.add_argument("--high_distillation", default = 0.5)                           # High level distillation factor (default: 0.5)

parser.add_argument("--train_max_episode", default = 500, type = int)               # number of episodes to train
parser.add_argument("--train_max_step", default = 730, type = int)                  # number of steps in each episode
parser.add_argument("--val_max_episode", default = 1, type=int)                     # number of episode to evaluate
parser.add_argument("--val_max_step", default = 200, type=int)                      # number of steps in each episode
parser.add_argument("--hidden_size", nargs = 2, default = [512, 256], type = tuple)  # default: [512, 256]
parser.add_argument("--batch_size", default = 256, type = int)
parser.add_argument("--replay_size", default = 10000, type = int)
parser.add_argument("--gamma", default = 0.99)
parser.add_argument("--tau", default = 0.001)
args = parser.parse_args()


# load data
with h5py.File('data\history_stock_price_us_22.h5','r') as f:
    history_stock_price = f['stock_price'][...]
    timestamp = [s.decode('utf-8') if type(s) == bytes else s for s in f['timestamp']]
    abbreviations = [s.decode('utf-8') if type(s) == bytes else s for s in f['abbreviations']]
    features = [s.decode('utf-8') if type(s) == bytes else s for s in f['features']]

train_step = timestamp.index('2017-06-29') # 2017-06-29
valid_step = timestamp.index('2019-07-01') # 2019-07-01

history_stock_price_training = history_stock_price[:,:train_step,:]
history_stock_price_validating = history_stock_price[:,train_step:valid_step,:]
history_stock_price_testing = history_stock_price[:,valid_step:,:]

timestamp_training = timestamp[:train_step]
timestamp_validating = timestamp[train_step:valid_step]
timestamp_testing = timestamp[valid_step:]

# Create the env
env_training = PortfolioEnvMain(history = history_stock_price_training,
                                abbreviation = abbreviations,
                                timestamp = timestamp_training,
                                window_length = args.window_size,
                                steps = args.train_max_step,
                                step_size = args.step_size,
                                feature_num = args.feature_num,
                                beta = 0.0,
                                name = args.env)

env_validating = PortfolioEnvMain(history = history_stock_price_validating,
                                  abbreviation = abbreviations,
                                  timestamp = timestamp_validating,
                                  window_length = args.window_size,
                                  steps = args.val_max_step,
                                  step_size = args.step_size,
                                  feature_num = args.feature_num,
                                  beta = 0.0,
                                  valid_env = True,
                                  name = args.env)

normal_state, oracle_state, groud_truth_state = env_training.reset()
state_dim = normal_state.flatten().shape[0]
action_dim = env_training.action_space.shape[0]
print("State dim:{}, Action dim:{}".format(state_dim, action_dim))

# Initialize replay memory
memory_l = ReplayMemory(int(args.replay_size))
memory_h = ReplayMemory(int(args.replay_size))

# Create model
agent = GEMS(state_dim = state_dim, 
             action_dim = action_dim, 
             hidden_size = args.hidden_size,
             gamma = args.gamma, 
             tau = args.tau, 
             ldf = args.low_distillation, 
             hdf = args.high_distillation)


best_reward = -np.inf
saved_reward = -np.inf
saved_ep = 0

for episode in range(args.train_max_episode):
    episode_reward = 0.0
    low_episode_value_loss = 0.0
    low_episode_policy_loss = 0.0
    low_episode_distillation_loss = 0.0
    high_episode_value_loss = 0.0
    high_episode_policy_loss = 0.0
    high_episode_distillation_loss = 0.0

    normal_state, oracle_state, groud_truth_state = env_training.reset()

    normal_state = normal_state.transpose((1,0,2))
    normal_state = normal_state.reshape(normal_state.shape[0],-1)
    normal_state = torch.Tensor([normal_state]).to(device)

    oracle_state = oracle_state.transpose((1,0,2))
    oracle_state = oracle_state.reshape(oracle_state.shape[0],-1)
    oracle_state = torch.Tensor([oracle_state]).to(device)

    # High-level policy: normal_state -> subgoal
    h_mu, h_sigma, subgoal = agent.get_subgoal(normal_state)

    for step in range(args.train_max_step):
        print("| Episode: {} | Step: {} | Last model saved with reward: {:.4f} at episode {}.".format(episode, step, saved_reward, saved_ep))
        
        # Low-level policy: normal_state + sub-goal -> action
        l_mu, l_sigma, action = agent.get_action(normal_state, subgoal)

        next_normal_state, next_oracle_state, reward, done, _, weights= env_training.step(action.cpu().numpy()[0])

        next_normal_state = next_normal_state.transpose((1,0,2))
        next_normal_state = next_normal_state.reshape(next_normal_state.shape[0],-1)
        next_normal_state = torch.Tensor([next_normal_state]).to(device)

        next_oracle_state = next_oracle_state.transpose((1,0,2))
        next_oracle_state = next_oracle_state.reshape(next_oracle_state.shape[0],-1)
        next_oracle_state = torch.Tensor([next_oracle_state]).to(device)

        # compute reward for low level policy
        cosine_reward = cosine_distance(subgoal.cpu().numpy()[0], weights)

        done = torch.Tensor([int(done)]).to(device)
        reward = torch.Tensor([reward]).to(device)
        cosine_reward = torch.Tensor([cosine_reward]).to(device)

        # get next subgoal
        next_h_mu, next_h_sigma, next_subgoal = agent.get_subgoal(next_normal_state)

        # Collect transitions
        memory_l.pushL(normal_state, oracle_state, next_normal_state, next_oracle_state, action, cosine_reward, done, subgoal, next_subgoal)
        memory_h.pushH(normal_state, oracle_state, next_normal_state, next_oracle_state, reward, done, subgoal)

        # Update observation
        normal_state = next_normal_state
        oracle_state = next_oracle_state

        # Update networks
        if len(memory_l) > args.batch_size:
            transitions_l = memory_l.sample(args.batch_size)
            batch_l = TransitionL(*zip(*transitions_l))
            value_loss_l, policy_loss_l, distillation_loss_l = agent.update_low_params(batch_l)

            low_episode_value_loss += value_loss_l
            low_episode_policy_loss += policy_loss_l
            low_episode_distillation_loss += distillation_loss_l

        if len(memory_h) > args.batch_size:
            transitions_h = memory_h.sample(args.batch_size)
            batch_h = TransitionH(*zip(*transitions_h))
            value_loss_h, policy_loss_h, distillation_loss = agent.update_high_params(batch_h)

            high_episode_value_loss += value_loss_h
            high_episode_policy_loss += policy_loss_h
            high_episode_distillation_loss += distillation_loss

        episode_reward += reward

        if done:
            print(done)
            break

    # On validation set
    for e in range(args.val_max_episode):
        val_high_episode_reward = 0.0
        val_low_episode_reward = 0.0

        s_state, o_state, ground_truth_obs = env_validating.reset()

        s_state = s_state.transpose((1,0,2))
        s_state = s_state.reshape(s_state.shape[0],-1)
        s_state = torch.Tensor([s_state]).to(device)


        # High-level policy: normal_state -> subgoal
        h_mu, h_sigma, subgoal = agent.get_subgoal(s_state)

        for s in range(args.val_max_step):
            _mu, _sigma, action = agent.get_action(s_state, subgoal)  # Selection without noise

            s_next_state, o_next_state, reward, done, _, weights= env_validating.step(action.cpu().numpy()[0])
            
            s_next_state = s_next_state.transpose((1,0,2))
            s_next_state = s_next_state.reshape(s_next_state.shape[0],-1)
            
            val_high_episode_reward += reward
            cosine_reward = cosine_distance(subgoal.cpu().numpy()[0], weights)
            val_low_episode_reward += cosine_reward

            s_next_state = torch.Tensor([s_next_state]).to(device)

            # get next subgoal
            next_h_mu, next_h_sigma, next_subgoal = agent.get_subgoal(s_next_state)
            s_state = s_next_state
            subgoal = next_subgoal

            if done:
                break

    # Save the model
    if val_high_episode_reward > best_reward:
        torch.save(agent.l_policy.state_dict(), '1111UnderHE_change_r.pkl')
        best_reward = val_high_episode_reward
        saved_reward = val_high_episode_reward
        saved_ep = episode + 1
