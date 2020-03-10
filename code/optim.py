"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: numpy array giving the current weights
  - dw: numpy array of the same shape as w giving the gradient of the
    loss with respect to w
  - config: dictionary containing hyperparameter values such as learning
    rate, momentum, etc; if the update rule requires caching values over many
    iterations, then config will also hold these cached values

Returns:
  - next_w: next point after the update
  - config: config dictionary to be passed to the next iteration of the
    update rule

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.

Code based on
    Stanford CS231n Convolutional Neural Networks for Visual Recognition
    http://cs231n.stanford.edu, http://cs231n.github.io
"""

import numpy as np


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
      - learning_rate: scalar learning rate
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
      - learning_rate: scalar learning rate
      - beta1: decay rate for moving average of first moment of gradient
      - beta2: decay rate for moving average of second moment of gradient
      - epsilon: small scalar used for smoothing to avoid dividing by zero
      - m: moving average of gradient
      - v: moving average of squared gradient
      - t: iteration number

    Reference:
        D. Kingma and J. Ba
        Adam: A Method for Stochastic Optimization
        ICLR 2015 (arXiv:1412.6980)
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    config['t'] += 1
    config['m'] = config['beta1']*config['m'] + (1 - config['beta1'])*dw
    config['v'] = config['beta2']*config['v'] + (1 - config['beta2'])*dw**2
    mhat = config['m'] / (1 - config['beta1']**config['t'])
    vhat = config['v'] / (1 - config['beta2']**config['t'])
    next_w = w - config['learning_rate']*mhat/(np.sqrt(vhat) + config['epsilon'])

    return next_w, config
