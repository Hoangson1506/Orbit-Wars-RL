from abc import ABC, abstractmethod
import torch    
import torch.nn as nn

class BaseAgent(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def act(self, obs):
        pass