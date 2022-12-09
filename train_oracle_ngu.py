import h5py
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from model.ddpg import DDPG
from network.rnd import RND
from env.portofolio import PortfolioEnvMain
from utils.replay_memory import ReplayMemory, Transition
from utils.sim_hash import SimHash
from utils.run_mv import RunningMV

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

def compute_rnd_error(predict_feature, target_feature):
        return (target_feature - predict_feature).pow(2).sum(1) / 2

def compute_episodic_reward(episodic_memory, current_c_state, k=20, kernel_cluster_distance=0.008,
                             kernel_epsilon=0.0001, c=0.001, sm=8) -> float:
    state_dist = [(c_state, torch.dist(c_state, current_c_state)) for c_state in episodic_memory]
    state_dist.sort(key=lambda x: x[1])
    state_dist = state_dist[:k]
    dist = [d[1].item() for d in state_dist]
    dist = np.array(dist)

    dist = dist / np.mean(dist)

    dist = np.max(dist - kernel_cluster_distance, 0)
    kernel = kernel_epsilon / (dist + kernel_epsilon)
    s = np.sqrt(np.sum(kernel)) + c

    if np.isnan(s) or s > sm:
        return 0
    return 1 / s

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

# Load data
with h5py.File('data\history_stock_price_cn_22.h5','r') as f:
    history_stock_price = f['stock_price'][...]
    timestamp = [s.decode('utf-8') if type(s) == bytes else s for s in f['timestamp']]
    abbreviations = [s.decode('utf-8') if type(s) == bytes else s for s in f['abbreviations']]
    features = [s.decode('utf-8') if type(s) == bytes else s for s in f['features']]

names = ['CASH', 'CMCSA.O', 'ADBE.O', 'GOOGL.O', 'AAPL.O', 
         'BRK_B.N', 'T.N', 'PG.N', 'XOM.N', 'DIS.N', 'UNH.N', 
         'JPM.N', 'CSCO.O', 'HD.N', 'AMZN.O', 'CRM.N', 'JNJ.N', 
         'KO.N', 'NFLX.O', 'VZ.N', 'MSFT.O', 'BAC.N', 'ABT.N']

train_step = timestamp.index('2017-06-29') # 2017-06-29
valid_step = timestamp.index('2019-07-01') # 2019-07-01

history_stock_price_training = history_stock_price[:,:train_step,:]
history_stock_price_validating = history_stock_price[:,train_step:valid_step,:]
history_stock_price_testing = history_stock_price[:,valid_step:,:]

timestamp_training = timestamp[:train_step]
timestamp_validating = timestamp[train_step:valid_step]
timestamp_testing = timestamp[valid_step:]

if __name__ == "__main__":
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

    state, ground_truth_obs = env_training.reset()
    state_dim = state.flatten().shape[0]
    action_dim = env_training.action_space.shape[0]
    print("State dim:{}, Action dim:{}".format(state_dim, action_dim))

    # Set random seed for all used libraries
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Define and build DDPG agent
    agent = DDPG(gamma = args.gamma,
                 tau = args.tau,
                 hidden_size = tuple(args.hidden_size),
                 state_dim = state_dim,
                 action_dim = action_dim,
                 )

    # Initialize RND module
    rnd = RND(state_dim=state_dim, action_dim=action_dim).to(device)
    rnd_optimizer = torch.optim.Adam(rnd.parameters(), lr=1e-3)
    forward_mse = torch.nn.MSELoss(reduction='none')

    # Initialize hash module
    hash_module = SimHash(obs_processed_flat_dim = state_dim + action_dim)

    # Initialize runningMV module
    long_run_mv = RunningMV()

    # Initialize replay memory
    memory = ReplayMemory(int(args.replay_size))

    # Define other variables
    best_reward = -np.inf
    saved_reward = -np.inf
    saved_ep = 0
    step = 0

    for episode in range(args.train_max_episode):
        episode_return = 0
        episode_value_loss = 0
        episode_policy_loss = 0
        episode_forward_loss = 0

        state, ground_truth_obs = env_training.reset()
        state = normalize_obs_diff(state)
        state = state.transpose((1,0,2))
        state = state.reshape(state.shape[0],-1)
        state = torch.Tensor([state])

        episodic_memory = []
        for s in range(args.train_max_step):
            step += 1
            print("| Episode: {} | Step: {} | Last model saved with reward: {:.4f} at episode {}.".format(episode, s, saved_reward, saved_ep))
            # interact with env
            _mu, _sigma, action = agent.calc_action(state)
            next_state, ext_reward, done, _, weights= env_training.step(action.cpu().numpy()[0])

            next_state = normalize_obs_diff(next_state)
            next_state = next_state.transpose((1,0,2))
            next_state = next_state.reshape(next_state.shape[0],-1)
            next_state = torch.Tensor([next_state])

            # compute rnd reward
            predict_feature, target_feature = rnd(state.to(device), action)
            rnd_error = compute_rnd_error(predict_feature, target_feature)
            long_run_mv.push(rnd_error)
            int_cross_reward = 1 + (rnd_error - long_run_mv.mean()) / (long_run_mv.variance() + 1e-9)
            int_cross_reward = torch.Tensor([int_cross_reward])
            # update rnd predictor's params
            rnd_optimizer.zero_grad()
            forward_loss = F.mse_loss(predict_feature, target_feature.detach())
            forward_loss.backward()
            rnd_optimizer.step()

            # compute intrinsic episodic reward
            combined_sw = np.concatenate((state.flatten().cpu().numpy(), weights.flatten()), 0)
            # weights = torch.Tensor([weights])
            # combined_sw = torch.cat((state.flatten(), weights.flatten()), 0)

            combined_sw = hash_module.compute_keys(combined_sw)
            combined_sw = torch.Tensor([combined_sw])
            episodic_memory.append(combined_sw)
            int_episodic_reward = compute_episodic_reward(episodic_memory, combined_sw)
            int_episodic_reward = torch.Tensor([int_episodic_reward])

            int_reward = int_cross_reward * int_episodic_reward
            episode_return += ext_reward

            mask = torch.Tensor([int(done)])
            ext_reward = torch.Tensor([ext_reward])

            memory.push(state, action, mask, next_state, ext_reward, int_reward)

            state = next_state

            if len(memory) > args.batch_size:
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))

                # Update actor and critic according to the batch
                value_loss, policy_loss = agent.update_params(batch)

            if done:
                break

        # On validation set
        for e in range(args.val_max_episode):
            val_episode_reward = 0

            state, ground_truth_obs = env_validating.reset()
            state = normalize_obs_diff(state)
            state = state.transpose((1,0,2))
            state = state.reshape(state.shape[0],-1)
            state = torch.Tensor([state]).to(device)

            for s in range(args.val_max_step):
                _mu, _sigma, action = agent.calc_action(state)  # Selection without noise

                next_state, ext_reward, done, _, weights = env_validating.step(action.cpu().numpy()[0])
                next_state = normalize_obs_diff(next_state)
                next_state = next_state.transpose((1,0,2))
                next_state = next_state.reshape(next_state.shape[0],-1)
                next_state = torch.Tensor([next_state]).to(device)

                state = next_state

                val_episode_reward += ext_reward

                cnt = 0
                for w in weights:
                    cnt += 1

                if done:
                    break

        # Save the model
        if val_episode_reward > best_reward:
            torch.save(agent.actor.state_dict(), 'oracle_cn.pkl')
            best_reward = val_episode_reward
            saved_reward = val_episode_reward
            saved_ep = episode + 1
    
    print("Train finished!")





