import torch
import torch.nn as nn

class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_capsules, num_routing_iterations):
        super(CapsuleLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_capsules = num_capsules
        self.num_routing_iterations = num_routing_iterations
        self.W = nn.Parameter(torch.randn(num_capsules, input_dim, output_dim))

    def squash(self, tensor):
        squared_norm = (tensor ** 2).sum(-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        x = torch.stack([x @ w for w in self.W], dim=1)
        b = torch.zeros(*x.size()[:-1], device=x.device)
        for _ in range(self.num_routing_iterations):
            c = torch.softmax(b, dim=1)
            s = (c.unsqueeze(-1) * x).sum(dim=2)
            v = self.squash(s)
            b = b + (x * v.unsqueeze(2)).sum(dim=-1)
        return v
