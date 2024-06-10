import torch.nn as nn
from collections import namedtuple, deque
import random
import torch
import imageio
from src.utils.utils import pad_sequences
from types import SimpleNamespace


activation_functions = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
    "softmax": nn.Softmax()
}


class Embedding(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[128],
                 non_linear=False, activation='relu', last_layer_activation=None):
        super(Embedding, self).__init__()
        self.mlp = nn.ModuleList([])
        if not non_linear:
            activation = 'identity'
        if hidden_layers is None:
            self.mlp.append(nn.Linear(input_size, output_size))
        else:
            for i in range(len(hidden_layers)):
                if i == 0:
                    self.mlp.append(nn.Linear(input_size, hidden_layers[i]))
                    self.mlp.append(activation_functions[activation])
                else:
                    self.mlp.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                    self.mlp.append(activation_functions[activation])

            self.mlp.append(nn.Linear(hidden_layers[-1], output_size))

        if last_layer_activation is not None:
            self.mlp.append(activation_functions[last_layer_activation])

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, data):
        result = self.mlp(data)
        return result


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer(object):
    
    def __init__(self, capacity=10000, trajectory_wise_sampling=False, off_policy=True, num_processes=1):
        self.capacity = capacity
        self.trajectory_wise_sampling = trajectory_wise_sampling
        self.off_policy = off_policy
        self.num_processes = num_processes
        on_policy = not off_policy

        if off_policy:
            self.memory = deque([], maxlen=capacity)
        elif not trajectory_wise_sampling and on_policy:
            self.memory = list()
        elif trajectory_wise_sampling and on_policy:
            self.memory = [[] for _ in range(num_processes)]

        self.temp_trajectory = [[] for _ in range(num_processes)]

    def reset(self):
        assert not self.off_policy
        if self.trajectory_wise_sampling:
            self.memory = [[] for _ in range(self.num_processes)]
        else:
            self.memory = list()

    def push(self, state, action, next_state, reward, done):
        # TODO done=1 -> true or name it mask
        for i in range(self.num_processes):
            transition = Transition(state[i],
                                    action[i],
                                    next_state[i],
                                    reward[i], 
                                    done[i])
            
            if self.trajectory_wise_sampling:
                self.temp_trajectory[i].append(transition)
                if done:
                    self.memory.append(self.temp_trajectory[i])
            else:
                self.memory.append(transition)
    
    def sample(self, batch_size):
        # TODO
        if self.trajectory_wise_sampling and not self.off_policy:
            assert self.num_processes == batch_size

        if self.trajectory_wise_sampling:
            samples = random.sample(self.memory, batch_size)

            max_len = max(len(batch) for batch in samples)

            # Assuming each element in state, action, next_state, and reward can be represented as a tensor
            states = pad_sequences(
                [[transition.state for transition in batch] for batch in samples],
                max_len, dim=2, batch_size=batch_size)  # Update dim according to your actual data dimension

            actions = pad_sequences(
                [[transition.action for transition in batch] for batch in samples],
                max_len, dim=1, batch_size=batch_size)  # Assuming actions are scalar, adjust as needed

            next_states = pad_sequences(
                [[transition.next_state for transition in batch] for batch in samples],
                max_len, dim=2, batch_size=batch_size)  # Update dim according to your actual data dimension

            rewards = pad_sequences(
                [[transition.reward for transition in batch] for batch in samples],
                max_len, dim=1, batch_size=batch_size)  # Assuming rewards are scalar, adjust as needed

        else:
            # TODO add whether the next state is terminal or not here instead of q_learning.calculate_loss  
            samples = random.sample(self.memory, batch_size)
            batch = Transition(*zip(*samples))
            states = torch.stack(batch.state, dim=0)
            next_states = torch.stack(batch.next_state, dim=0)
            actions = torch.stack(batch.action, dim=0)
            rewards = torch.stack(batch.reward, dim=0).unsqueeze(-1).float()
            dones = torch.stack(batch.done, dim=0).float()
            batch = {'states': states,
                     'actions': actions,
                     'next_states': next_states,
                     'rewards': rewards,
                     'dones':dones}
            batch = SimpleNamespace(**batch)
        return batch

    def __len__(self):
        return len(self.memory)
