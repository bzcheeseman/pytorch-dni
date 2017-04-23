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
                 optimizer_kwargs={'lr': 0.001, 'weight_decay': 0},
                 delta_optimizer=optim.Adam,
                 delta_optimizer_kwargs={'lr': 0.001, 'weight_decay': 1e-6}
                 ):

        super(DNI, self).__init__()
        self.update_module = update_module
        self.labels = labels
        self.update = update

        if labels:  # can use single layer network
            self.synth_grad = nn.Linear(update_num_out + 1, update_num_out)
            self.synth_grad.weight.data.uniform_(-1e-6, 1e-6)
            self.synth_grad.bias.data.zero_()
        else:  # use 2-layer network in the case of no labels
            self.synth_grad = nn.Sequential(OrderedDict([
                ("lin1", nn.Linear(update_num_out, int(1.25*update_num_out))),  # upsize a little bit
                ("lin2", nn.Linear(int(1.25*update_num_out), update_num_out))
            ]))
            self.synth_grad[0].weight.data.uniform_(-1e6, 1e6)
            self.synth_grad[0].bias.data.zero_()
            self.synth_grad[1].weight.data.uniform_(-1e6, 1e6)
            self.synth_grad[1].bias.data.zero_()

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


# if __name__ == "__main__":
#
#     dni1 = DNI(nn.Sequential(nn.Linear(10, 6), nn.ReLU(), nn.Linear(6, 4)), 4)
#     dni2 = DNI(nn.Linear(4, 3), 3)
#     out = DNI(nn.Linear(3, 3), 3, update=False)
#
#     criterion = nn.CrossEntropyLoss()
#
#     out_optimizer = out.init_optimizer(optim.Adam, {'lr': 0.001})
#
#     out_optimizer.zero_grad()
#
#     input = Variable(torch.rand(4, 10))
#     label = Variable(torch.LongTensor(4).random_(0, 2))
#
#     loss = []
#     for i in range(int(5e3)):
#         output1 = dni1(input)  # update 1
#
#         output2 = dni2(output1)  # update 2
#
#         dni1.synthetic_update(dni2.update_grad)
#
#         output3 = out(output2)
#         l = criterion(output3, label)
#         l.backward()
#         out_optimizer.step()
#
#         dni2.synthetic_update(out.update_grad)
#
#         loss.append(l.data[0])
#
#     import matplotlib.pyplot as plt
#
#     plt.plot(loss)
#     plt.show()







