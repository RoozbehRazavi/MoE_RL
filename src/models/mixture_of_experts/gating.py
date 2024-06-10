import torch.nn as nn
from src.models.utils import Embedding


class ActionWiseGatingNetwork(nn.Module):
    pass


class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_outputs, hidden_layers=(64,), activation='relu', sparse=False, topk=0):
        super(GatingNetwork, self).__init__()
        self.num_outputs = num_outputs
        self.sparse = sparse
        self.topk = topk

        self.gating = Embedding(input_size, num_outputs,
                                hidden_layers=hidden_layers, non_linear=True, activation=activation,
                                last_layer_activation='softmax')

    def forward(self, data):
        result = self.gating(data)
        return result

