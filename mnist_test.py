#
# Created by Aman LaChapelle on 4/22/17.
#
# pytorch-dni
# Copyright (c) 2017 Aman LaChapelle
# Full license at pytorch-dni/LICENSE.txt
#

import torchvision as tv
from model import DNI
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as Funct
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from tensorboard_logger import configure, log_value

configure("training/first_test_mnist_labels_wd")


class MNISTExtractor(nn.Module):
    def __init__(self):
        super(MNISTExtractor, self).__init__()

        self.FE = nn.Sequential(
            nn.Conv2d(1, 8, 7),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.num_outs = 16 * ((((28 - 7) + 1) - 3) + 1)**2

    def forward(self, x):
        x = self.FE(x)
        return x


if __name__ == "__main__":
    transform = tv.transforms.ToTensor()

    batch = 25

    train = tv.datasets.MNIST("data", train=True, transform=transform, download=False)
    train_loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=4)

    test = tv.datasets.MNIST("data", train=False, transform=transform, download=False)
    test_loader = DataLoader(test, batch_size=batch, shuffle=True, num_workers=4)

    mne = MNISTExtractor()
    classifier = nn.Linear(mne.num_outs, 10)

    dni_mne = DNI(mne, mne.num_outs, labels=True, update=True)
    dni_classifier = DNI(classifier, 10, update=False)  # need last layer to be dni also, just set update=False

    classifier_optimizer = dni_classifier.init_optimizer(optim.Adam, {'lr': 0.001})  # others are dealt with
    criterion = nn.CrossEntropyLoss()

    print_steps = 5

    for epoch in range(1):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs)
            labels = Variable(labels)

            features = dni_mne(inputs, labels)

            features = features.view(features.size(0), -1)
            outputs = dni_classifier(features)
            l = criterion(outputs, labels)
            l.backward()
            classifier_optimizer.step()

            dni_mne.synthetic_update(dni_classifier.update_grad)

            running_loss += l.data[0]

            if i % print_steps == print_steps-1:
                current_step = i + 1 + len(train_loader) * epoch
                print("Current step: ", current_step, "Loss: ", running_loss / print_steps)
                log_value("Loss", running_loss / print_steps, step=current_step)

                running_loss = 0.0

    print("Training Finished")
    dni_mne.update = False
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs)
        labels = Variable(labels)

        features = dni_mne(inputs)
        features = features.view(features.size(0), -1)
        outputs = dni_classifier(features)
        l = criterion(outputs, labels)
        total_loss += l.data[0]

        _, max_index = np.where(outputs.cpu().data.numpy() == np.max(outputs.cpu().data.numpy()))
        if max_index[0] == labels.data[0]:
            correct += 1

    print("Test avg loss: ", total_loss/len(test_loader))
    print("Percent correct: ", correct/len(test_loader) * 100)



