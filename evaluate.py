from src.util_vis.evaluate import *
from src.models.reinforcement_learning.q_network import *
import argparse
import gymnasium as gym
import yaml
from stable_baselines3.dqn.dqn import DQN
from src.util_vis.utils import save_config_to_yaml, load_config_from_yaml, setup_rl_logger, recreate_directory
from box import Box
from src.models.utils import TimeStepReplayBuffer
from src.algorithm.q_learning import SparsePolicyMixtureDQN

# TODO make it minimal
def evaluate(exp_path, env_name, gamma, logs_path):
    parser = argparse.ArgumentParser(description='Reload the configurations.')

    # Add initial arguments
    parser.add_argument('--config_file', type=str, default='PacMan_DQN.yaml', help='Path to the configuration file.')
    parser.add_argument('--model_c', type=str, choices=['model1', 'model2', 'model3', 'model4'], help='Choose a DL model to use.', default='model1')#, required=True)
    parser.add_argument('--exp_name', type=str, help='Experiment name.', default='test')#, required=True)
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
    parser.add_argument('--model_c', type=str, choices=['model1', 'model2', 'model3', 'model4'], help='Choose a DL model to use.', default='model1')#, required=True)
    parser.add_argument('--exp_name', type=str, help='Experiment name.', default='test')#, required=True)
    parser.add_argument('--channel_wise_input', type=bool)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='List of seeds for running experiments.')

    # Final parse of arguments, with defaults potentially overridden by command line inputs
    args = parser.parse_args()

    env = gym.make(env_name, render_mode='rgb_array')
    model = SparsePolicyMixtureDQN(SparseMixtureDQNPolicy, env, 
                                   buffer_size=10000,
                                   batch_size=args.training.batch_size,
                                   verbose=1,
                                   policy_kwargs={'mixture': args},
                                   replay_buffer_class=TimeStepReplayBuffer, 
                                   replay_buffer_kwargs={})
    
    agent = model.load(exp_path)

    info = evaluate_model(agent, env, gamma, device='cuda')


    plot_kernel(agent, path=logs_path, name='kernel') 
    plot_weights_channel(info['channel_gatings'], path=logs_path, name='highest_channel_gating')
    plot_weight_channel_anim(info['channel_gatings'], path=logs_path, name='expert_gating')
    assess_value_functions_diversity(info['experts_values'])
    plot_value_functions_diversity(info['experts_values'], path=logs_path, name='value_function_diversity')
    plot_weights(info['experts_gating'], path=logs_path, name='expert_gating')
    visualize_policy_functions(info['experts_values'], path=logs_path, name='policy_maximum_action')
    visualize_value_functions(info['experts_values'], path=logs_path, name='value_maximum_action')
    plot_trajectory(info['rendered'], path=logs_path, name='behavior')




evaluate(
        './experiments2/16_sparsity/seed_0/16_sparsity.zip',
        'MsPacman-v4', 0.99, logs_path='./logs7')