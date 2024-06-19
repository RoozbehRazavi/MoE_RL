import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_wei_lin(data):
    return

def plot_durations(episode_durations, str_, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(durations_t.cpu().numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.cpu().numpy())

    plt.savefig(f'./{str_}/res.png')

def plot_wei(weights, path, name):
    plt.figure(figsize=(10, 6)) 
    plt.imshow(weights, cmap='viridis', aspect='auto', interpolation='nearest')

    plt.title('Attention Weights of Modules over Time')
    plt.xlabel('Experts')
    plt.ylabel('Time Steps')
    cbar = plt.colorbar()
    cbar.set_label('Attention Weight')

    plt.savefig(f'{path}/{name}.png')

    plt.close()

def plot_tsne(weights, states, itr, str_, name):
    num_time_steps = len(states)

    max_module_index = np.argmax(weights, axis=0)

    # Perform t-SNE embedding on the states to reduce their dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embedded_states = tsne.fit_transform(states)

    # Plot the t-SNE embedded states, coloring each point based on the index of the module with the highest weight
    plt.figure(figsize=(10, 6))
    for i in range(num_time_steps):
        plt.scatter(embedded_states[i, 0], embedded_states[i, 1], c=max_module_index[i], cmap='tab10')

    plt.colorbar(label='Module Index')
    plt.title('t-SNE Embedded States colored by Module Index with Highest Weight')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # Save the plot without displaying it
    plt.savefig(f'./{str_}/tSNE_states_{itr}.png')  # Specify the file name and format

    # Close the plot to free up memory
    plt.close()
