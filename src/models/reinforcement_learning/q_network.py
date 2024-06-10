from typing import Any, Dict, List, Type, Union
import torch
from gymnasium.spaces.discrete import Discrete
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
import torch.nn as nn
from torch.optim.adam import Adam as Adam
from torch.optim.optimizer import Optimizer as Optimizer
from ..utils import Embedding
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.dqn.policies import DQNPolicy
from ..mixture_of_experts.mixture_of_experts import PolicyMixtureOfExperts
from src.models.perception.cnn_encoder import LunarLanderCNN
from gymnasium import spaces
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from src.models.mixture_of_experts.mixture_of_experts import MixtureOfExperts
from src.models.perception.cnn_encoder import CustomCNN
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor
)
    
def constant_fn(val: float) -> Schedule:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: constant value
    :return: Constant schedule function.
    """

    def func(_):
        return val

    return func


def get_schedule_fn(value_schedule: Union[Schedule, float]) -> Schedule:
    """
    Transform (if needed) learning rate and clip range (for PPO)
    to callable.

    :param value_schedule: Constant value of schedule function
    :return: Schedule function (can return constant value)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constant_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    # Cast to float to avoid unpickling errors to enable weights_only=True, see GH#1900
    # Some types are have odd behaviors when part of a Schedule, like numpy floats
    return lambda progress_remaining: float(value_schedule(progress_remaining))


class MixtureQNetwork(QNetwork):
    def __init__(self, *args, mixture, state_size, **kwargs) -> None:
        super().__init__(**kwargs)
        action_size = self.action_space.n
        self.q_net = PolicyMixtureOfExperts(mixture, state_size, action_size)


class MixtureDQNPolicy(DQNPolicy):
    def __init__(self, *args, mixture, **kwargs) -> None:
        self.mixture = mixture
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> MixtureQNetwork:
        encoder = LunarLanderCNN(self.net_args['observation_space'],
                                 mixture=self.mixture)
        net_args = self._update_features_extractor(self.net_args, features_extractor=encoder)
        state_size = encoder.convw * encoder.convh
        return MixtureQNetwork(mixture=self.mixture, state_size=state_size, **net_args).to(self.device)


class EmbeddingMixtureQNetwork(QNetwork):
    def __init__(self, *args, mixture, state_size, **kwargs) -> None:
        super().__init__(**kwargs)
        action_size = self.action_space.n
        self.q_net = MixtureOfExperts(mixture, state_size, action_size)


class EmbeddingMixtureDQNPolicy(DQNPolicy):
    def __init__(self, *args, mixture, **kwargs) -> None:
        self.mixture = mixture
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> EmbeddingMixtureQNetwork:
        encoder = LunarLanderCNN(self.net_args['observation_space'],
                                 mixture=self.mixture)
        net_args = self._update_features_extractor(self.net_args, features_extractor=encoder)
        state_size = encoder.convw * encoder.convh
        return EmbeddingMixtureQNetwork(mixture=self.mixture, state_size=state_size, **net_args).to(self.device)





