import numpy as np
import layers


class PolicyNet(object):
    """
    Policy network consisting of two dense layers.
    """

    def __init__(self, input_dim, num_actions, nhidden=100):
        """
        Initialize a new network.
        """
        # initialize weights and biases
        self.params = {}
        self.params['W1'] = np.random.randn(nhidden, input_dim) / np.sqrt(input_dim)
        self.params['b1'] = np.zeros(nhidden)
        self.params['W2'] = np.random.randn(num_actions, nhidden) / np.sqrt(nhidden)
        self.params['b2'] = np.zeros(num_actions)


    def evaluate(self, x):
        """
        Evaluate the network for input 'x', returning softmax probability distribution.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        h, _      = layers.affine_relu_forward(x, W1, b1)
        logits, _ = layers.affine_forward(h, W2, b2)
        return layers.softmax(logits)


    def loss(self, x, y, weights):
        """
        Evaluate softmax loss (maximum likelihood estimation)
        scaled for each sample i by 'weights[i]',
        and corresponding gradients.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        h,      cache1 = layers.affine_relu_forward(x, W1, b1)
        logits, cache2 = layers.affine_forward(h, W2, b2)

        loss, dloss = layers.weighted_softmax_loss(logits, y, weights)

        # backpropagate
        dx2, dw2, db2 = layers.affine_backpropagate(dloss, cache2)
        dx1, dw1, db1 = layers.affine_relu_backpropagate(dx2, cache1)
        grads = {}
        grads['W1'] = dw1
        grads['W2'] = dw2
        grads['b1'] = db1
        grads['b2'] = db2

        return loss, grads
