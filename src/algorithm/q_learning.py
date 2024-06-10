# import torch.nn as nn
# import random
# import torch
# import numpy as np
# from ..models.reinforcement_learning.q_network import Q1Network
# from ..models.perception.cnn_encoder import LunarLanderCNN
# from ..models.mixture_of_experts.mixture_of_experts import MixtureOfExperts, PolicyMixtureOfExperts
# import math


class Agent(object):
    def __init__(self) -> None:
        pass

    def calculate_loss(self, batch):
        pass

    def select_action(self, state):
        pass

    def after_update(self, iteration):
        pass

    def learnable_params(self):
        pass


class QLearning(nn.Module, Agent):
    def __init__(self, args, state_size, action_size, device='cuda') -> None:
        super(QLearning, self).__init__()
        channel = args.model.state_encoder.state_embedding.layers.channel
        encoder = LunarLanderCNN(channel)
        if state_size is None:
            state_size = encoder.convh * encoder.convw * channel
        if not args.model.distrbuted_value:
            self.q_net = Q1Network(state_size, action_size)
            self.target_q_net = Q1Network(state_size, action_size)
            if args.model.state_encoder.type == 'cnn':
                self.state_encoding = encoder
            elif args.model.state_encoder.type == 'cnn_moe':
                self.state_encoding = nn.Sequential(encoder,
                                                    MixtureOfExperts(
                                                        args, encoder.convh * encoder.convw))
            else:
                raise Exception()
        else:
            print('Mixture of policies')
            print(state_size)
            self.q_net = PolicyMixtureOfExperts(args, state_size, output_dim=action_size,)
            self.target_q_net = PolicyMixtureOfExperts(args, state_size, output_dim=action_size)
            if args.model.state_encoder.type == 'cnn':
                self.state_encoding = encoder

        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.steps_done = 0
        self.action_size = action_size
        self.state_size = state_size
        self.eps_threshold = None
        self.update_target_every = args.training.update_target_every
        self.update_target_softly = args.training.update_target_softly
        self.TAU = args.training.TAU
        self.device = device
        self.batch_size = args.training.batch_size
        self.gamma = args.training.gamma
        self.eps_threshold = args.training.eps_start
        self.args = args

    def forward(self, states, target=False):
        batch_size = states.shape[0]
        states = states.permute(0, -1, 1, 2)
        # TODO states, gating_output, extra, out_gat, entropy = self.state_encoding(states)
        states = self.state_encoding(states).reshape(batch_size, -1)
        if target:
            with torch.no_grad():
                return self.target_q_net(states)#, gating_output, extra, out_gat, entropy
        return self.q_net(states)#, gating_output, extra, out_gat, entropy

    def calculate_loss(self, batch):
        dones = 1 - batch.dones
        non_final_mask = dones.to(torch.bool).squeeze(-1).squeeze(-1).to(self.device)
        non_final_next_states = batch.next_states[non_final_mask]
        state_batch = batch.states
        action_batch = batch.actions
        reward_batch = batch.rewards

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # state_action_values, _, _, _, _ = self(state_batch)
        state_action_values = self(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # tmp, _, _, _, _ = self(non_final_next_states, True)
            tmp = self(non_final_next_states, True)
            next_state_values[non_final_mask] = tmp.max(1).values

        next_state_values = next_state_values.unsqueeze(-1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        if torch.isnan(loss):
            print('op')
        return loss

    def select_action(self, state, random_=True):
        sample = random.random()
        if sample > self.eps_threshold or not random_:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # TODO tmp, gating_weight, extra, out_gat, _ = self(state)
                tmp = self(state)
                return tmp.max(1).indices.view(1, 1) #, gating_weight, extra, out_gat
        else:
            return torch.tensor([[np.random.randint(0, self.action_size)]], device=self.device, dtype=torch.long)#, None, None, None

    def after_update(self, iteration):
        if iteration % self.update_target_every == 0:
            if self.update_target_softly:
                target_net_state_dict = self.target_q_net.state_dict()
                policy_net_state_dict = self.q_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                                1 - self.TAU)
                self.target_q_net.load_state_dict(target_net_state_dict)
            else:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

        if self.args.training.update_eps:
            if iteration % self.args.training.update_eps_every == 0:
                self.eps_threshold = self.args.training.end_eps + (
                            self.args.training.eps_start - self.args.training.end_eps) * \
                                     math.exp(-1. * self.steps_done / self.args.training.eps_decay)

    def learnable_params(self):
        return [*self.q_net.parameters(), *self.state_encoding.parameters()]
