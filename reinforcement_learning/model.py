
import numpy as np
import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 416):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        extractors = {}
        total_concat_size = 0
        
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        for key, subspace in observation_space.spaces.items():
            if key == "main":
                n_input_channels = subspace.shape[0]
                extractors[key] = nn.Sequential(
                        nn.BatchNorm2d(n_input_channels),
                        nn.Conv2d(n_input_channels, 12, kernel_size=5, stride=(1, 2)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(12, 32, kernel_size=5, stride=(1, 2)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 44, kernel_size=5, stride=(1, 2)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(44, 32, kernel_size=3, stride=(1, 2)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 16, kernel_size=3, stride=(1, 2)),
                        nn.ReLU(inplace=True),
                        nn.Flatten()
                    )
                total_concat_size += 1280
                
            if key == 'HP':
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(4, 16),
                    nn.ReLU(inplace=True),
                    nn.Linear(16, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 512)
                )
                total_concat_size += 512
            
                
            # if key == 'item':
            #     extractors[key] = nn.Sequential(
            #         nn.Linear(subspace.shape[0], 4),
            #         nn.Linear(4, 1)
            #     )
            
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = 512

        self.linear1 = nn.Sequential(nn.Linear(total_concat_size, 1024), 
                                    nn.ReLU(),
                                    nn.Linear(1024, 512), 
                                    nn.ReLU()
                                    )
        
        self.start = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))      
        x = self.linear1(torch.cat(encoded_tensor_list, dim=1))
        return x