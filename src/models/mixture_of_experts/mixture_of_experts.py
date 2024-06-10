import torch.nn as nn
import torch
from .experts import Expert, MemoryDQNExperts, DQNExperts
from .gating import GatingNetwork, ActionWiseGatingNetwork
from ..utils import Embedding
import copy
from src.models.mixture_of_experts.experts import Q1Network
import concurrent.futures
from torch.nn.parallel import parallel_apply


class PolicyMixtureOfExperts(nn.Module):
    def __init__(self, args, state_size, action_size):
        super(PolicyMixtureOfExperts, self).__init__()
        input_dim = state_size
        output_dim = action_size
        self.input_dim = input_dim #it should be w*h
        self.output_dim = output_dim
        self.num_experts = args.model.MOE.num_experts
        self.channel_wise_input = args.model.MOE.channel_wise_input
        self.channel = args.model.state_encoder.state_embedding.layers.channel

        if self.channel_wise_input:
            experts_input_dim = input_dim 
        else:
            experts_input_dim = input_dim * self.channel
        if args.model.MOE.memory_expert:
            self.experts = nn.ModuleList([MemoryDQNExperts(experts_input_dim, output_dim, self.channel, self.channel_wise_input) for _ in range(self.num_experts)])
        else:
            # TODO input size should be different with respect to channel_wise_input
            self.experts = nn.ModuleList([DQNExperts(experts_input_dim, output_dim, self.channel, self.channel_wise_input) for _ in range(self.num_experts)])

        self.gating = GatingNetwork(input_dim * self.channel, self.num_experts)


    def forward(self, data):
        data = data.reshape(data.shape[0], self.channel, self.input_dim)
        batch_size = data.shape[0]
        experts_output = [self.experts[i](data) for i in range(self.num_experts)]
        experts_output = torch.stack([x for x in experts_output], dim=1)
        gating_output = self.gating(data.reshape(batch_size, -1)).unsqueeze(-1)
        experts_output = gating_output * experts_output
        experts_output = experts_output.sum(dim=1).reshape(batch_size, -1)
        return experts_output


class MixtureOfExperts(nn.Module):
    def __init__(self, args, input_dim, action_size):

        super(MixtureOfExperts, self).__init__()

        assert args.model.MOE.outer_recursion_threshold is not None or not args.model.MOE.outer_recursive
        assert not args.model.MOE.outer_recursive or (1 > args.model.MOE.outer_recursion_threshold > 0)

        self.outer_recursive = args.model.MOE.outer_recursive
        self.outer_recursion_threshold = args.model.MOE.outer_recursion_threshold
        self.MAX_OUT_REC = args.model.MOE.MAX_OUT_REC
        self.input_dim = input_dim
        self.output_dim = args.model.MOE.experts_output_dim
        self.num_experts = args.model.MOE.num_experts
        channel = args.model.state_encoder.state_embedding.layers.channel
        self.channel = channel if args.model.MOE.channel_wise_input else 1

        self.experts = nn.ModuleList([Expert(args=args, input_size=self.input_dim) for _ in range(self.num_experts)])
        self.gating = GatingNetwork(self.input_dim, self.num_experts)
        self.outer_recursive_gating = GatingNetwork(self.output_dim * self.channel, 2)

        self.recursive_mapping = Embedding(self.output_dim, self.input_dim)

        self.outer_recursive = args.model.MOE.outer_recursive

        self.q_network = Q1Network(args.model.MOE.experts_output_dim * self.channel, action_size)

    def _forward(self, data):
        batch_size = data.shape[0]
        data_ = data.reshape(batch_size * self.channel, -1)
        experts_output = [expert(data_) for expert in self.experts]
        gating_counter = [x[1] for x in experts_output]
        total_gating_counter = torch.stack(gating_counter).sum(-1).reshape(self.num_experts, 1)
        experts_output = torch.stack([x[0] for x in experts_output]).permute(1, 0, 2)
        gating_output = self.gating(data_).unsqueeze(-1)
        entropy = -(gating_output * torch.log(gating_output)).sum(1).mean(0)
        experts_output = gating_output * experts_output
        experts_output = experts_output.sum(dim=1).reshape(batch_size, -1)

        return experts_output, gating_output.reshape(batch_size, self.channel, self.num_experts).sum(1).reshape(batch_size, self.num_experts).detach(),  total_gating_counter.detach()/data.shape[1], entropy

    def forward(self, data):
        batch_size = data.shape[0]
        result, gatting_weights, gatting_index, entropy = self._forward(data)
        inner_gating_co = self.outer_recursive_gating(result)
        gating2_temp = (inner_gating_co[:, 0] > self.outer_recursion_threshold).int().unsqueeze(-1)
        total_gat_out = copy.deepcopy(gating2_temp.squeeze(-1).detach().cpu())
        counter = 0

        if self.outer_recursive:
            while gating2_temp.sum() > 0:
                if counter >= self.MAX_OUT_REC:
                    break
                result_ = result.reshape(batch_size * self.channel, self.output_dim)
                # TODO
                result_ = self.recursive_mapping(result_)
                result_ = result_.reshape(batch_size, self.channel, self.input_dim)
                output, gatting_weights_, gatting_index_, entropy = self._forward(result_)

                gatting_weights += gatting_weights_
                gatting_index += gatting_index_
                #data = output * inner_gating_co[:, 0].unsqueeze(-1)  * gating2_temp + x * (1 - gating2_temp)
                result = result + output * inner_gating_co[:, 0].unsqueeze(-1) * gating2_temp #+ x * (1 - gating2_temp)

                inner_gating_co = self.outer_recursive_gating(result)
                gating2_temp = (inner_gating_co[:, 0] > 0.5).int().unsqueeze(-1)
                total_gat_out += gating2_temp.squeeze(-1).detach().cpu()
                counter = counter + 1

        values = self.q_network(result)
        return values
        # gatting_weights/(counter+1), gatting_index/(counter+1), total_gat_out, entropy

