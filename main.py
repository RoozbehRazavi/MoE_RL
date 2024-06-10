import argparse
import gymnasium as gym
import torch.cuda
from box import Box
from src.utils.utils import save_config_to_yaml, load_config_from_yaml, setup_rl_logger, recreate_directory
#from src.scripts.trainer import Trainer
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.dqn.dqn import DQN
from src.models.reinforcement_learning.q_network import MixtureDQNPolicy
from src.models.reinforcement_learning.q_network import EmbeddingMixtureDQNPolicy
from stable_baselines3.common.torch_layers import NatureCNN
from src.models.perception.cnn_encoder import LunarLanderCNN
from src.models.perception.cnn_encoder import CustomCNN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import imageio


def main():
    parser = argparse.ArgumentParser(description='Reload the configurations.')
    parser.add_argument('--config_file', default='PacMan_DQN.yaml', type=str)
    args = parser.parse_args()

    config = load_config_from_yaml(f'./configs/{args.config_file}')
    config = Box(config)
    parser = argparse.ArgumentParser(description='Reload the configurations.')
    parser.set_defaults(**config)
    args = parser.parse_args()
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    args.path.experiment_path = f'./experiments2/{args.experiment_name}/{date}'
    args.path.save_path = f'{args.path.experiment_path}/{args.path.save_path}'
    args.path.figures_path = f'{args.path.experiment_path}/{args.path.figures_path}'
    args.path.videos_path = f'{args.path.experiment_path}/{args.path.videos_path}'
    args.path.logs_path = f'{args.path.experiment_path}/{args.path.logs_path}'
    recreate_directory(args.path.experiment_path)
    save_config_to_yaml(args, f'{args.path.experiment_path}/configs.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seeds = args.training.seeds
    writer = SummaryWriter(args.path.logs_path) #setup_rl_logger(args.path.logs_path)
    returns = 0

    for seed in seeds:
        trainer = Trainer(args, device, seed, writer)
        returns += trainer.train()

    returns /= len(args.training.seeds)
    print(returns)
    # logger.info('Batch logging for episode', {'episode': 1, 'returns': returns})


class CustomTensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomTensorBoardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here L2 norm of weights)
        all_weights = np.concatenate([p.data.cpu().numpy().flatten() for p in self.model.policy.parameters()])
        l2_norm = np.sqrt(np.sum(np.square(all_weights)))
        self.logger.record('custom/l2_norm', l2_norm)
        return True


def make_env():
    def _init():
        env = gym.make("MsPacman-v4", render_mode='rgb_array')
        return env
    return _init


def render(model, env, path, episode=1):
    frames = []
    for episode in range(episode):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            frame = env.render()  # Render the environment
            frames.append(frame)

    imageio.mimsave(f'{path}', frames, fps=30)

import argparse
import gymnasium as gym
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from box import Box
from tensorboard.backend.event_processing import event_accumulator
from memory_profiler import profile, LogFile

# Your custom policy classes (replace with actual imports)
# from your_policy_module import MixtureDQNPolicy, CustomCNN, EmbeddingMixtureDQNPolicy

def load_config_from_yaml(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def read_returns(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    # Print all available tags
    print(f"Available tags in {log_dir}: {ea.Tags()}")
    returns = [s.value for s in ea.Scalars('rollout/ep_rew_mean')]
    return returns

import random
import numpy as np
import torch

def set_seed(seed):
    # Set seed for random module
    random.seed(seed)
    
    # Set seed for numpy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

def main1():
    parser = argparse.ArgumentParser(description='Reload the configurations.')

    # Add initial arguments
    parser.add_argument('--config_file', type=str, default='PacMan_DQN.yaml', help='Path to the configuration file.')
    parser.add_argument('--model_c', type=str, choices=['model1', 'model2', 'model3'], help='Choose a DL model to use.', required=True)
    parser.add_argument('--exp_name', type=str, help='Experiment name.', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='List of seeds for running experiments.')

    # Parse known arguments, allowing for further arguments in the config
    args, unknown = parser.parse_known_args()

    # Load configuration from a YAML file
    config = load_config_from_yaml(f'./configs/{args.config_file}')
    config = Box(config)

    # Create a new parser for re-parsing with defaults from the configuration
    parser = argparse.ArgumentParser(description='Reload the configurations.')

    # Set defaults from the loaded configuration
    parser.set_defaults(**config)

    # Re-add the same command line arguments to allow for overriding
    parser.add_argument('--config_file', type=str, default='PacMan_DQN.yaml', help='Path to the configuration file.')
    parser.add_argument('--model_c', type=str, choices=['model1', 'model2', 'model3'], help='Choose a DL model to use.', required=True)
    parser.add_argument('--exp_name', type=str, help='Experiment name.', required=True)
    parser.add_argument('--channel_wise_input', type=bool)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='List of seeds for running experiments.')

    # Final parse of arguments, with defaults potentially overridden by command line inputs
    args = parser.parse_args()

    print('args are ready!')

    log_dirs = [f"./experiments2/{args.exp_name}/seed_{seed}" for seed in args.seeds]

    for seed, log_dir in zip(args.seeds, log_dirs):
        torch.cuda.empty_cache()
        set_seed(seed)
        env = gym.make("MsPacman-v4", render_mode='rgb_array')
        env = Monitor(env)  # Wrap the environment in a Monitor

        # Ensure the environment's action space is supported
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(f"The algorithm only supports Discrete action spaces but {type(env.action_space)} was provided.")

        if args.model_c == 'model1':
            model = DQN(MixtureDQNPolicy, env,
                        buffer_size=10000,
                        batch_size=args.training.batch_size, verbose=1, tensorboard_log=log_dir, policy_kwargs={'mixture': args})
        elif args.model_c == 'model2':
            model = DQN('MlpPolicy', env,
                        buffer_size=10000,
                         policy_kwargs={
                             'features_extractor_class': CustomCNN,
                             'features_extractor_kwargs': {'features_dim': 500},
                            'net_arch': [11184, 11184]},
                         verbose=1,
                         tensorboard_log=log_dir)
        elif args.model_c == 'model3':
            model = DQN(EmbeddingMixtureDQNPolicy, env, verbose=1,
                        buffer_size=10000,
                        tensorboard_log=log_dir, policy_kwargs={'mixture': args})

        total_params1 = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        print(f"{args.model_c}: {total_params1}")
        total_timesteps = 2e6
        model.learn(total_timesteps=total_timesteps, log_interval=100)
        model.save(f'{log_dir}/{args.exp_name}')
        env.close()

    # Aggregate results and log to TensorBoard
    all_returns = []
    log_dirs = [f"./experiments2/{args.exp_name}/seed_{seed}/DQN_1" for seed in args.seeds]
    for log_dir in log_dirs:
        returns = read_returns(log_dir)
        all_returns.append(returns)

    min_length = min(map(len, all_returns))
    all_returns = [returns[:min_length] for returns in all_returns]  # Truncate to the same length
    mean_returns = np.mean(all_returns, axis=0)

    aggregated_log_dir = f"./experiments2/{args.exp_name}/aggregated"
    writer = SummaryWriter(aggregated_log_dir)
    for i, mean_return in enumerate(mean_returns):
        writer.add_scalar('average_return', mean_return, i)
    writer.close()


if __name__ == "__main__":
    main1()






