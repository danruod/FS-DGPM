import sys
import torch
from torch import nn
from torch.nn import functional as F

class Learner(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_tasks, args):
        super(Learner, self).__init__()

        # n_inputs: ch * img_size * img_size
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_tasks = n_tasks

        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(self.n_inputs, 100, bias=False)
        self.fc2 = torch.nn.Linear(100, 100, bias=False)
        self.head = torch.nn.Linear(100, self.n_outputs, bias=False)

        # number of representation matrix
        self.n_rep = 3
        self.multi_head = False

    def forward(self, x, vars=None, svd=False):
        h = x.reshape(x.size(0), -1)
        if svd:
            y = []
            y.append(h)
            h = self.relu(self.fc1(h))
            y.append(h)
            h = self.relu(self.fc2(h))
            y.append(h)
        elif vars is not None:
            assert len(vars) == 3
            h = self.relu(F.linear(h, vars[0]))
            h = self.relu(F.linear(h, vars[1]))
            y = F.linear(h, vars[2])
        else:
            h = self.relu(self.fc1(h))
            h = self.relu(self.fc2(h))
            y = self.head(h)

        return y

    def get_params(self):
        self.vars = []
        for p in list(self.parameters()):
            if p.requires_grad:
                self.vars.append(p)
        return self.vars
