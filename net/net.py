import torch.nn as nn
import torch
from .fcnn import FullyCnn


class FlatteningLayer(nn.Module):
    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class EmbeddingNetwork(nn.Module):
    def __init__(self, input_channels=1, emb_dim=64, input_size=64):
        super(EmbeddingNetwork, self).__init__()
        fully_cnn = FullyCnn(input_channels=1)
        flattener = FlatteningLayer()
        fake_input = torch.zeros((1, input_channels, input_size, input_size))
        output_tensor = fully_cnn.forward(fake_input)
        output_tensor = flattener.forward(output_tensor)
        fully_connected = nn.Sequential(nn.Linear(output_tensor.shape[-1], emb_dim))
        self.network = nn.Sequential(fully_cnn, flattener, fully_connected)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))

    def forward(self, input_tensor):
        return self.network(input_tensor)
