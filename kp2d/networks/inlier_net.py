# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
import torch.nn.functional as F

# Slightly modified version of 1d-CNN from https://arxiv.org/abs/1905.04132.
# More details: https://github.com/vislearn/ngransac
# Code adapted from https://github.com/vislearn/ngransac/blob/master/network.py

class InlierNet(torch.nn.Module):
    def __init__(self, blocks):
        super(InlierNet, self).__init__()

        self.res_blocks = []
        self.bn_momentum = 0.1
        self.p_in = torch.nn.Sequential(torch.nn.Conv2d(5, 128, 1, 1, 0, bias=False),
                                        torch.nn.BatchNorm2d(128, momentum=0.9))
        for i in range(0, blocks):
            self.res_blocks.append((
                torch.nn.Conv2d(128, 128, 1, 1, 0),
                torch.nn.BatchNorm2d(128, momentum=self.bn_momentum),
                torch.nn.Conv2d(128, 128, 1, 1, 0),
                torch.nn.BatchNorm2d(128, momentum=self.bn_momentum),
                ))

        # register list of residual block with the module
        for i, r in enumerate(self.res_blocks):
            super(InlierNet, self).add_module(str(i) + 's0', r[0])
            super(InlierNet, self).add_module(str(i) + 's1', r[1])
            super(InlierNet, self).add_module(str(i) + 's2', r[2])
            super(InlierNet, self).add_module(str(i) + 's3', r[3])

        # output are 1D sampling weights (log probabilities)
        self.p_out = torch.nn.Conv2d(128, 1, 1, 1, 0)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.p_in(x))

        for r in self.res_blocks:
            res = x
            x = F.relu(r[1](F.instance_norm(r[0](x))))
            x = F.relu(r[3](F.instance_norm(r[2](x))))
            x = x + res
        return self.p_out(x)
