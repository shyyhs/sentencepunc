import torch
import torch.nn as nn

class LinearDecoder(nn.Module):
    def __init__(self, args):
        super(LinearDecoder, self).__init__()

        self.input_size = args.hidden_size
        self.layers = nn.Sequential(
                nn.Linear(self.input_size, 2),
                )

        #self.layers = nn.Sequential(
        #        nn.Linear(self.input_size, 256),
        #        nn.Tanh(),
        #        nn.Linear(256, 64),
        #        nn.Tanh(),
        #        nn.Linear(64,2),
        #        )

    def forward(self, inputs):
        output = self.layers(inputs)
        return output
