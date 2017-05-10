#
# Created by Aman LaChapelle on 4/22/17.
#
# pytorch-dni
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-dni/LICENSE.txt
#

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as Funct
import torch.optim as optim
from collections import OrderedDict


class DNI(nn.Module):
    def __init__(self,
                 update_module,
                 update_num_out,
                 labels=False,
                 update=True,
                 optimizer=optim.Adam,
                 optimizer_kwargs={'lr': 0.001, 'weight_decay': 1e-9},
                 delta_optimizer=optim.Adam,
                 delta_optimizer_kwargs={'lr': 0.001, 'weight_decay': 1e-6}
                 ):

        super(DNI, self).__init__()
        self.update_module = update_module
        self.labels = labels
        self.update = update

        if labels:  # can use single layer network <- change this so we can initialize it outside the network...
            self.synth_grad = nn.Sequential(
                nn.Linear(update_num_out + 1, update_num_out),
                nn.PReLU(update_num_out)
            )
            self.synth_grad.weight.data.uniform_(-1e-6, 1e-6)
            self.synth_grad.bias.data.zero_()
        else:  # use 2-layer network in the case of no labels
            self.synth_grad = nn.Sequential(
                nn.Linear(update_num_out, update_num_out),
                nn.PReLU(update_num_out),
                nn.Linear(update_num_out, update_num_out),
                nn.PReLU(update_num_out)
            )
            self.synth_grad[0].weight.data.uniform_(-1e-6, 1e-6)
            self.synth_grad[0].bias.data.zero_()
            self.synth_grad[2].weight.data.uniform_(-1e-6, 1e-6)
            self.synth_grad[2].bias.data.zero_()

        self.optimizer = optimizer(self.update_module.parameters(), **optimizer_kwargs)

        self.delta_optimizer = delta_optimizer(self.synth_grad.parameters(), **delta_optimizer_kwargs)
        self.delta_criterion = nn.L1Loss()

        self.update_grad = Variable(torch.FloatTensor([0]))
        self.delta = Variable(torch.FloatTensor([0]), requires_grad=True)

    def init_optimizer(self, optimizer, optimizer_kwargs):
        if self.update:
            return
        else:
            optimizer = optimizer(self.update_module.parameters(), **optimizer_kwargs)
            return optimizer

    def forward(self, x, label=None):

        self.optimizer.zero_grad()
        self.delta_optimizer.zero_grad()

        def save_grad(grad):
            self.update_grad = grad
            self.update_grad.volatile = False
            return grad

        x.detach_()
        x.requires_grad = True
        hook = x.register_hook(save_grad)

        x = self.update_module(x)

        if self.update:
            if label:
                self.delta = self.synth_grad(torch.cat([x.view(x.size(0), -1), label.unsqueeze(1).float()], 1))
                self.delta = self.delta.view(x.size())
            else:
                self.delta = self.synth_grad(x.view(x.size(0), -1))
                self.delta = self.delta.view(x.size())

            x.backward(self.delta.data, retain_variables=True)
            self.optimizer.step()
            hook.remove()

        return x

    def synthetic_update(self, next_layer_grad):
        loss = self.delta_criterion(self.delta, next_layer_grad)
        loss.backward()
        self.delta_optimizer.step()


class CNNDNI(nn.Module):
    def __init__(self,
                 update_module,
                 update_out_filters,
                 labels=False,
                 update=True,
                 optimizer=optim.Adam,
                 optimizer_kwargs={'lr': 0.001, 'weight_decay': 0},
                 delta_optimizer=optim.Adam,
                 delta_optimizer_kwargs={'lr': 0.001, 'weight_decay': 0}
                 ):
        super(CNNDNI, self).__init__()

        self.update_module = update_module
        self.labels = labels
        self.update = update

        if labels:
            self.synth_grad = nn.Sequential(
                nn.Conv2d(update_out_filters + 1, update_out_filters, 3, padding=1),
                nn.BatchNorm2d(update_out_filters),
                nn.PReLU(update_out_filters)
            )
        else:
            self.synth_grad = nn.Sequential(
                nn.Conv2d(update_out_filters, update_out_filters, 3, padding=1),
                nn.BatchNorm2d(update_out_filters),
                nn.PReLU(update_out_filters)
            )

        self.optimizer = optimizer(self.update_module.parameters(), **optimizer_kwargs)

        self.delta_optimizer = delta_optimizer(self.synth_grad.parameters(), **delta_optimizer_kwargs)
        self.delta_criterion = nn.L1Loss()

        self.update_grad = Variable(torch.FloatTensor([0]))
        self.delta = Variable(torch.FloatTensor([0]), requires_grad=True)

    def init_optimizer(self, optimizer, optimizer_kwargs):
        if self.update:
            return
        else:
            optimizer = optimizer(self.update_module.parameters(), **optimizer_kwargs)
            return optimizer

    def forward(self, x, label=None):

        self.optimizer.zero_grad()

        def save_grad(grad):
            self.update_grad = grad
            self.update_grad.volatile = False
            return grad

        x.detach_()
        x.requires_grad = True
        hook = x.register_hook(save_grad)

        x = self.update_module(x)

        if self.update:
            if label:
                label = label.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, x.size(2), x.size(3))
                self.delta = self.synth_grad(torch.cat([x, label.float()], 1))
            else:
                self.delta = self.synth_grad(x)

            x.backward(self.delta.data, retain_variables=True)
            self.optimizer.step()
            hook.remove()

        return x

    def synthetic_update(self, next_layer_grad):
        self.delta_optimizer.zero_grad()
        loss = self.delta_criterion(self.delta, next_layer_grad)
        loss.backward()
        self.delta_optimizer.step()




