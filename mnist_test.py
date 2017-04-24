#
# Created by Aman LaChapelle on 4/22/17.
#
# pytorch-dni
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-dni/LICENSE.txt
#

import torchvision as tv
from model import DNI, CNNDNI
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as Funct
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from tensorboard_logger import configure, log_value

configure("training/cnndni_9_cuda_3")


class MNISTExtractor(nn.Module):
    def __init__(self):
        super(MNISTExtractor, self).__init__()

        self.update = True

        self.FE1 = CNNDNI(nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        ), update_out_filters=16, labels=True, update=self.update)
        self.FE2 = CNNDNI(nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.Tanh()
        ), update_out_filters=32, labels=True, update=self.update)

        self.num_outs = int(32 * ((np.floor(((28 - 5) + 1)/2) - 3) + 1)**2)
        self.classifier = DNI(nn.Linear(self.num_outs, 10), update_num_out=10, update=False)
        self.class_opt = self.classifier.init_optimizer(optim.Adam, {'lr': 0.001})

    def set_update_false(self):
        self.update = False
        for mod in self.modules():
            try:
                mod.update = False
            except AttributeError:
                pass

    def forward(self, x, labels=None):
        x = self.FE1(x, labels)  # update 1
        x = self.FE2(x, labels)  # update 2

        if self.update:
            self.FE1.synthetic_update(self.FE2.update_grad.cuda())  # update 3

        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # return the output
        return x

    def optimizer_step(self, outputs, labels, criterion):
        l = criterion(outputs, labels)
        l.backward()
        self.class_opt.step()  # update 4

        self.FE2.synthetic_update(self.classifier.update_grad.cuda())  # update 5

        return l

if __name__ == "__main__":
    transform = tv.transforms.ToTensor()

    log_file = open("training_log.txt", 'w+')

    batch = 13
    synth_grad_update = 1

    train = tv.datasets.MNIST("data", train=True, transform=transform, download=False)
    train_loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=4)

    test = tv.datasets.MNIST("data", train=False, transform=transform, download=False)
    test_loader = DataLoader(test, batch_size=batch, num_workers=4)

    dni_mne = MNISTExtractor()
    dni_mne.cuda()
    criterion = nn.CrossEntropyLoss()

    print_steps = 100

    for epoch in range(1000):  # 52 epochs at batch size 250
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs)
            labels = Variable(labels)

            for j in range(synth_grad_update):
                outputs = dni_mne(inputs.cuda(), labels.cuda())

            l = dni_mne.optimizer_step(outputs, labels.cuda(), criterion)

            running_loss += l.data[0]

            if i % print_steps == print_steps-1:
                current_step = i + 1 + len(train_loader) * epoch
                log_file.write("Current step: {} Loss: {}\n".format(current_step, running_loss / print_steps))
                # log_value("Loss", running_loss / print_steps, step=current_step)
                running_loss = 0.0

        torch.save(dni_mne.state_dict(), "models/mnist_model_ckpt_{}.dat".format(epoch))

    print("Training Finished")
    torch.save(dni_mne.state_dict(), "models/mnist_model.dat")
    # import os
    # os.remove("models/mnist_model_ckpt*")

    dni_mne.set_update_false()
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs)
        labels = Variable(labels)

        outputs = dni_mne(inputs.cuda())
        # outputs = classifier(features)
        l = criterion(outputs, labels.cuda())
        total_loss += l.data[0]

        _, max_index = np.where(outputs.cpu().data.numpy() == np.max(outputs.cpu().data.numpy()))
        if max_index[0] == labels.data[0]:
            correct += 1

    print("Test avg loss: ", total_loss / len(test_loader))
    print("Percent correct: ", (correct / len(test_loader)) * 100)
    # no higher than 13%-15% max previously (DNI better than baseline)



