# pytorch-dni

Working on implementing DNI from this [paper](https://arxiv.org/abs/1608.05343) from DeepMind
in pytorch.

Took lots of inspiration from the [torch implementation](https://github.com/kbullaughey/dni-synthetic-gradients)

Notes:
1. This will work better than it does currently on larger networks
2. It doesn't seem to be nearly as effective as full backprop on simple tasks (which makes sense)
3. I'm not getting the learning speedup for some reason..

Help is always welcome!