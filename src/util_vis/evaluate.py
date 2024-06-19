import torch
from src.util_vis.utils import save_frames_as_video
from src.util_vis.plotter import plot_wei, plot_tsne, plot_durations, plot_wei_lin
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def evaluate_model(agent, env, GAMMA, device):
    done = False
    state, _ = env.reset()
    return_ = 0
    rendereds = []
    #states = []
    values = []
    channel_gatings = []
    experts_gating = []
    experts_values = []
    counter = 0
    with torch.no_grad():
        while not done:
            rendered = env.render()#.transpose(2, 0, 1)
            rendereds.append(rendered)
            
            #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            #states.append(state)
            
            action, _states = agent.predict(state, deterministic=True)
            
            q_values, info = agent.policy.q_net(torch.permute(torch.tensor(state, device=device), (-1, 0, 1)).unsqueeze(0))

            
            values.append(q_values)
            channel_gatings.append(info['channel_wise_gating'])
            experts_gating.append(info['expertes_gating'])
            experts_values.append(info['experts_values'])
            
            state, reward, done, _, _ = env.step(action.item())
            return_ += (reward * (GAMMA ** counter))
            counter += 1
    
    channel_gatings = torch.cat(channel_gatings, dim=0).cpu().numpy()
    experts_gating = torch.cat(experts_gating, dim=0).cpu().numpy()
    #states = torch.cat(states, dim=0).cpu().numpy()
    values = torch.cat(values, dim=0).cpu().numpy()
    experts_values = torch.cat(experts_values, dim=0).cpu().numpy()
    # plot_tsne(weights, states, itr, fig)
    info = {
        'states': rendereds,
        'channel_gatings': channel_gatings, 
        'experts_gating': experts_gating,
        'experts_values': experts_values,
        'values': values, 
        'rendered': rendered,
        'return': return_
    }
    return info

def plot_weights_channel(weights, path, name):
    number_experts = weights.shape[1]
    time_steps = weights.shape[0]
    channels = weights.shape[2]
    
    # Create a single plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for experts
    colors = plt.cm.viridis(np.linspace(0, 1, number_experts))

    for i in range(number_experts):
        # Get the index of the maximum weight channel at each time step for the i-th expert
        max_channel_indices = np.argmax(weights[:, i, :, 0], axis=1)
        
        # Plot each expert's data in the same axes, each in a different color
        ax.plot(range(time_steps), max_channel_indices, color=colors[i], label=f'Expert {i+1}')

    ax.set_title('Highest Weight Channel Over Time by Expert')
    ax.set_xticks(range(time_steps))
    ax.set_yticks(range(channels))
    ax.set_yticklabels([f'Channel {j+1}' for j in range(channels)])
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Channel')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{path}/{name}.png')  # Save as PNG file

def plot_weight_channel_anim(weights, path, name):
    number_experts = weights.shape[1]
    time_steps = weights.shape[0]
    channels = weights.shape[2]
    # Setup the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(weights[0, :, :, 0], aspect='auto', interpolation='nearest', cmap='viridis')

    # Set up the colorbar and titles
    fig.colorbar(heatmap, ax=ax)
    ax.set_title('Time Step 0')
    ax.set_xlabel('Channels')
    ax.set_ylabel('Experts')
    ax.set_xticks(range(channels))
    ax.set_yticks(range(number_experts))
    ax.set_xticklabels([f'Channel {i+1}' for i in range(channels)])
    ax.set_yticklabels([f'Expert {i+1}' for i in range(number_experts)])

    # Animation update function
    def update(frame):
        ax.set_title(f'Time Step {frame}')
        heatmap.set_data(weights[frame, :, :, 0])

    # Create animation
    ani = FuncAnimation(fig, update, frames=time_steps, repeat=False)

    # To display in a Jupyter notebook:
    # from IPython.display import HTML
    # HTML(ani.to_jshtml())

    # To save as a video file
    ani.save(f'{path}/{name}.mp4', writer='ffmpeg', fps=10)
    plt.close()  # Close the figure to prevent static plot from showing in the notebook



def plot_kernel(agent, path, name):
    slope = agent.q_net.experts_learnable_kernel.slope
    bias = agent.q_net.experts_learnable_kernel.bias
    mag_sin = agent.q_net.experts_learnable_kernel.mag_sin
    mag_cos = agent.q_net.experts_learnable_kernel.mag_cos
    phase = agent.q_net.experts_learnable_kernel.phase
    freq = agent.q_net.experts_learnable_kernel.freq
    x = torch.arange(-50, 50).to('cuda')

    #plt.figure(figsize=(15, 5))
    for i in range(agent.q_net.experts_learnable_kernel.num_experts):
        # Compute the function values
        # f = (slope[i] * x + bias[i]).squeeze(0)
        # g = (mag_cos[i] * torch.cos(freq[i] * x + phase[i]) + mag_sin[i] * torch.sin(freq[i] * x + phase[i])).squeeze(0)
        # sigmoid_matrix = torch.sigmoid(f + g)
        sigmoid_matrix = torch.tanh(agent.q_net.experts_learnable_kernel.steepness * torch.sin(freq[i] * (x + bias[i]))).squeeze(0)
        # Plotting f
        # plt.subplot(1, 3, 1)
        # plt.plot(x.cpu().detach().numpy(), f.cpu().detach().numpy(), label=f'Expert {i+1} f')
        # plt.title('Linear Component f')
        # plt.xlabel('x')
        # plt.ylabel('f(x)')

        # # Plotting g
        # plt.subplot(1, 3, 2)
        # plt.plot(x.cpu().detach().numpy(), g.cpu().detach().numpy(), label=f'Expert {i+1} g')
        # plt.title('Non-linear Component g')
        # plt.xlabel('x')
        # plt.ylabel('g(x)')

        # Plotting the sigmoid matrix
        #plt.subplot(1, 3, 3)
        plt.plot(x.cpu().detach().numpy(), sigmoid_matrix.cpu().detach().numpy(), label=f'Expert {i+1} Sigmoid')
        plt.title('Sigmoid Function of the Combined Outputs')
        plt.xlabel('x')
        plt.ylabel('Sigmoid Output')

    # Adding legends to each subplot
    # plt.subplot(1, 3, 1)
    # plt.legend()
    # plt.subplot(1, 3, 2)
    # plt.legend()
    # plt.subplot(1, 3, 3)
    # plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(f'{path}/{name}.png') 


def assess_value_functions_diversity(value_functions_outputs):
    # value_functions_outputs should be a numpy array of shape (length, number_models, action_size)
    length, number_models, action_size = value_functions_outputs.shape
    
    # Calculate variance across models for each action at each timestep
    variance_per_timestep_action = np.var(value_functions_outputs, axis=1)
    mean_variance_per_action = np.mean(variance_per_timestep_action, axis=0)
    
    # Aggregate mean variance across all actions
    overall_mean_variance = np.mean(mean_variance_per_action)
    
    # Calculate pairwise distances for each action across all models
    mean_pairwise_distances_per_action = []
    for action in range(action_size):
        # Reshape to (length, number_models) for each action
        model_outputs_for_action = value_functions_outputs[:, :, action]
        # Compute pairwise distances using Euclidean distance
        pairwise_distances = squareform(pdist(model_outputs_for_action.T, 'euclidean'))
        mean_pairwise_distance = np.mean(pairwise_distances)
        mean_pairwise_distances_per_action.append(mean_pairwise_distance)
    
    overall_mean_pairwise_distance = np.mean(mean_pairwise_distances_per_action)
    
    print("Mean Variance Across Actions and Timesteps:", overall_mean_variance)
    print("Mean Pairwise Distance Between Models for Each Action:", overall_mean_pairwise_distance)

    return variance_per_timestep_action, mean_pairwise_distances_per_action

def plot_value_functions_diversity(value_functions_outputs, path='./logs2', name='value_diversity'):
    # value_functions_outputs should be a numpy array of shape (length, number_models, action_size)
    length, number_models, action_size = value_functions_outputs.shape
    
    # Calculate variance across models for each action at each timestep
    variance_per_timestep_action = np.var(value_functions_outputs, axis=1)
    
    # Prepare to plot variance
    plt.figure(figsize=(14, 7))
    for action in range(action_size):
        plt.plot(variance_per_timestep_action[:, action], label=f'Action {action}')
    plt.title('Variance of Q-values Across Models for Each Action')
    plt.xlabel('Time Step')
    plt.ylabel('Variance')
    plt.legend()
    plt.savefig(f'{path}/variance_{name}.png')

    # Calculate and plot pairwise distances for each action across all models
    plt.figure(figsize=(14, 7))
    #for time in range(length):
        # Reshape to (length, number_models) for each action
        #model_outputs_for_action = value_functions_outputs[time, :, :]
        # Compute pairwise distances using Euclidean distance
    pairwise_distances = np.var(value_functions_outputs, axis=(1, 2))#squareform(pdist(model_outputs_for_action.T, 'euclidean'))
        #mean_pairwise_distance = np.mean(pairwise_distances, axis=0)  # Mean over pairs per timestep
    plt.plot(pairwise_distances)
    plt.title('Mean Pairwise Euclidean Distance Between Models for Each Action')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Pairwise Distance')
    plt.legend()
    plt.savefig(f'{path}/distance_{name}.png')

def plot_weights(weights, path, name):
    plot_wei(weights, path, name)

def visualize_policy_functions(value_functions_outputs, path='./logs2', name='policy_experts'):
    # value_functions_outputs should be a numpy array of shape (length, num_models, action_size)
    length, num_models, action_size = value_functions_outputs.shape
    
    # Prepare to plot max Q-values
    plt.figure(figsize=(14, 7))
    
    # Iterate over each model
    for model_index in range(num_models):
        # Extract the max Q-value at each timestep for the current model
        max_q_values = np.max(value_functions_outputs[:, model_index, :], axis=1)
        
        # Plotting the max Q-values for this model
        plt.plot(max_q_values, label=f'Model {model_index}')
    
    plt.title('Maximum Q-values Over Time for Each Model')
    plt.xlabel('Time Step')
    plt.ylabel('Max Q-value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}/{name}.png')

def visualize_value_functions(value_functions_outputs, path='./logs2', name='value_experts'):
    length, num_models, action_size = value_functions_outputs.shape
    
    # Prepare to plot preferred actions
    plt.figure(figsize=(14, 7))
    
    # Iterate over each model
    for model_index in range(num_models):
        # Get the indices of the max Q-value action at each timestep for the current model
        preferred_actions = np.argmax(value_functions_outputs[:, model_index, :], axis=1)
        
        # Plotting the preferred actions for this model
        plt.plot(preferred_actions, label=f'Model {model_index}')
    
    plt.title('Preferred Actions Over Time for Each Model')
    plt.xlabel('Time Step')
    plt.ylabel('Action Index')
    plt.yticks(range(action_size), [f'Action {i}' for i in range(action_size)])  # Set tick labels to action indices
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}/{name}.png')

def plot_trajectory(rendereds, path='./logs2', name='behaviour'):
    save_frames_as_video(rendereds, path, name)
