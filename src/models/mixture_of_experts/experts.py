import torch.nn as nn
from src.models.utils import Embedding
from .gating import GatingNetwork
import copy
import torch
from collections import deque


class Q1Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Q1Network, self).__init__()
        self.network = Embedding(input_size=input_size,
                                 output_size=output_size,
                                 hidden_layers=(2048, 2048),
                                 non_linear=True)

    def forward(self, state):
        value = self.network(state)
        return value


class Expert(nn.Module):
    def __init__(self, args, input_size, hidden_layers=(128,), activation='relu', non_linear=True):
        super(Expert, self).__init__()

        assert args.model.MOE.in_recursion_threshold is not None or not args.model.MOE.in_recursive
        assert not args.model.MOE.in_recursive or (1 > args.model.MOE.in_recursion_threshold > 0)

        self.MAX_IN_REC = args.model.MOE.MAX_IN_REC
        self.recursion_threshold = args.model.MOE.in_recursion_threshold
        self.recursive = args.model.MOE.in_recursive
        self.input_size = input_size
        self.output_size = args.model.MOE.experts_output_dim
        self.mlp = Embedding(input_size=self.input_size, output_size=self.output_size,
                             hidden_layers=(2148, 2148), activation=activation, non_linear=non_linear)

        if self.recursive:
            self.inner_gating = GatingNetwork(input_size=self.output_size, num_outputs=2, hidden_layers=(64,))

            self.mapping_in_to_out = Embedding(input_size=self.output_size, output_size=self.input_size, non_linear=False)

    def _forward(self, data):
        result = self.mlp(data)
        return result

    def forward(self, data):
        result = self._forward(data)

        if not self.recursive:
            return result, torch.zeros((data.shape[0]))

        inner_gating_co = self.inner_gating(result)
        gating2_temp = (inner_gating_co[:, 0] > self.recursion_threshold).int().unsqueeze(-1)

        total_gat = copy.deepcopy(gating2_temp.squeeze(-1).detach().cpu())

        counter = 0
        while gating2_temp.sum() > 0:
            if counter >= self.MAX_IN_REC:
                break
            result_ = self.mapping_in_to_out(result)
            output = self._forward(result_)
            result = result + output * inner_gating_co[:, 0].unsqueeze(-1) * gating2_temp
            inner_gating_co = self.inner_gating(result)
            gating2_temp = (inner_gating_co[:, 0] > 0.25).int().unsqueeze(-1)
            total_gat += gating2_temp.squeeze(-1).detach().cpu()
            counter = counter + 1

        return result, total_gat


class DQNExperts(nn.Module):
    def __init__(self, input_size, output_size, channel, channel_wise):
        super(DQNExperts, self).__init__()
        self.channel = channel
        self.channel_wise = channel_wise
        self.action_size = output_size
        #breakpoint()
        # TODO now, alpha and beta are generated based on observations 
        if self.channel_wise:
            self.policy = Q1Network(input_size, output_size)
            self.channel_gating = GatingNetwork(input_size * channel, channel)
        else:
            self.policy = Q1Network(input_size, output_size)

    def forward(self, data):
        # data.shape: (batch_size, channel, w, h)
        batch_size = data.shape[0]
        if type(self.channel) is tuple:
            self.channel = self.channel[0]
        data = data.reshape(batch_size, self.channel, -1)
        if self.channel_wise:
            gating_data = data.reshape(batch_size, -1)
            gating = self.channel_gating(gating_data).reshape(batch_size, self.channel).unsqueeze(-1)
            expert_data = data.reshape(batch_size * self.channel, -1)
            values = self.policy(expert_data).reshape(batch_size, self.channel, self.action_size)
            values = values * gating
            values = values.sum(1).squeeze(1)
        else:
            expert_data = data.reshape(batch_size, -1)
            values = self.policy(expert_data).reshape(batch_size, self.action_size)

        return values


# Episodic Memory with Attention
class EpisodicMemory:
    def __init__(self):
        self.keys = deque(maxlen=100)
        self.values = deque(maxlen=100)

    def get_value(self, query):
        if not self.keys:  # if memory is empty, return zero
            return torch.tensor(0.0)
        keys_tensor = torch.stack(list(self.keys))
        query = query.repeat(keys_tensor.shape[0], 1)

        # Compute similarity scores using dot product
        attention_scores = torch.sum(keys_tensor * query, dim=1)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=0)

        # Compute weighted sum of values
        values_tensor = torch.tensor(list(self.values), dtype=torch.float32)
        weighted_sum = torch.sum(values_tensor * attention_weights)
        return weighted_sum

    def add_memory(self, key, value):
        self.keys.append(key)
        self.values.append(value)


class MemoryDQNExperts(nn.Module):
    def __init__(self, input_size, output_size):
        super(MemoryDQNExperts, self).__init__()
        self.policy = Q1Network(input_size, output_size)
        self.memory = EpisodicMemory()
        self.gating = GatingNetwork(input_size=input_size, num_outputs=2)

    def forward(self, data):
        value1 = self.policy(data)
        value2 = self.memory.get_value(data)
        weights = self.gating(data)
        value = value1 * weights[:, 0] + value2 * weights[:, 1]
        return value


