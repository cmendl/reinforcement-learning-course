"""
Unit tests for layers module.

Code inspired by
    Stanford CS231n Convolutional Neural Networks for Visual Recognition
    http://cs231n.stanford.edu, http://cs231n.github.io
"""

import unittest
import numpy as np
import layers


class TestLayers(unittest.TestCase):


    def test_affine_forward(self):

        num_inputs = 2
        input_shape = (4, 5, 6)
        output_dim = 3

        input_size  = num_inputs * np.prod(input_shape)
        weight_size = output_dim * np.prod(input_shape)

        x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
        w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim).T
        b = np.linspace(-0.3, 0.1, num=output_dim)

        out, _  = layers.affine_forward(x, w, b)

        out_ref = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                            [ 3.25553199,  3.5141327,   3.77273342]])

        self.assertAlmostEqual(rel_error(out, out_ref), 0., delta=1e-8,
                               msg='output of affine layer must agree with reference')


    def test_affine_backpropagate(self):

        batch_size = 10
        input_dim  = 7
        output_dim = 3

        x = np.random.randn(batch_size, input_dim)
        w = np.random.randn(output_dim, input_dim)
        b = np.random.randn(output_dim)
        dout = np.random.randn(batch_size, output_dim)

        # numeric gradients as reference
        dx_num = eval_numerical_gradient_array(lambda x: layers.affine_forward(x, w, b)[0], x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: layers.affine_forward(x, w, b)[0], w, dout)
        db_num = eval_numerical_gradient_array(lambda b: layers.affine_forward(x, w, b)[0], b, dout)

        _, cache = layers.affine_forward(x, w, b)
        dx, dw, db = layers.affine_backpropagate(dout, cache)

        self.assertAlmostEqual(rel_error(dx, dx_num), 0., delta=1e-7, msg='dx for affine layer must agree with numeric gradient')
        self.assertAlmostEqual(rel_error(dw, dw_num), 0., delta=1e-7, msg='dw for affine layer must agree with numeric gradient')
        self.assertAlmostEqual(rel_error(db, db_num), 0., delta=1e-7, msg='db for affine layer must agree with numeric gradient')


    def test_sigmoid_backpropagate(self):

        x = np.random.randn(10, 15)
        dout = np.random.randn(*x.shape)

        # numeric gradients as reference
        dx_num = eval_numerical_gradient_array(lambda x: layers.sigmoid_forward(x)[0], x, dout)

        _, cache = layers.sigmoid_forward(x)
        dx = layers.sigmoid_backpropagate(dout, cache)

        self.assertAlmostEqual(rel_error(dx, dx_num), 0., delta=1e-9, msg='dx for sigmoid layer must agree with numeric gradient')


    def test_relu_backpropagate(self):

        x = np.random.randn(12, 17)
        dout = np.random.randn(*x.shape)

        # numeric gradients as reference
        dx_num = eval_numerical_gradient_array(lambda x: layers.relu_forward(x)[0], x, dout)

        _, cache = layers.relu_forward(x)
        dx = layers.relu_backpropagate(dout, cache)

        self.assertAlmostEqual(rel_error(dx, dx_num), 0., delta=1e-9, msg='dx for ReLu layer must agree with numeric gradient')


    def test_conv_forward(self):

        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)

        conv_param = {'stride': 2, 'pad': 1}
        out, _ = layers.conv_forward(x, w, b, conv_param)

        out_ref = np.array([[[[-0.08759809, -0.10987781],
                              [-0.18387192, -0.2109216 ]],
                             [[ 0.21027089,  0.21661097],
                              [ 0.22847626,  0.23004637]],
                             [[ 0.50813986,  0.54309974],
                              [ 0.64082444,  0.67101435]]],
                            [[[-0.98053589, -1.03143541],
                              [-1.19128892, -1.24695841]],
                             [[ 0.69108355,  0.66880383],
                              [ 0.59480972,  0.56776003]],
                             [[ 2.36270298,  2.36904306],
                              [ 2.38090835,  2.38247847]]]])

        self.assertAlmostEqual(rel_error(out, out_ref), 0., delta=1e-7,
                           msg='output of convolutional layer must agree with reference')


    def test_conv_backpropagate(self):

        N = 4   # mini-batch size
        F = 5   # number of features
        C = 3   # number of channels

        x = np.random.randn(N, C, 6, 7)
        w = np.random.randn(F, C, 3, 2)
        b = np.random.randn(F,)
        dout = np.random.randn(N, F, 3, 4)
        conv_param = {'stride': 2, 'pad': 1}

        dx_num = eval_numerical_gradient_array(lambda x: layers.conv_forward(x, w, b, conv_param)[0], x, dout)
        dw_num = eval_numerical_gradient_array(lambda w: layers.conv_forward(x, w, b, conv_param)[0], w, dout)
        db_num = eval_numerical_gradient_array(lambda b: layers.conv_forward(x, w, b, conv_param)[0], b, dout)

        out, cache = layers.conv_forward(x, w, b, conv_param)
        dx, dw, db = layers.conv_backpropagate(dout, cache)

        self.assertAlmostEqual(rel_error(dx, dx_num), 0., delta=1e-7, msg='dx for conv layer must agree with numeric gradient')
        self.assertAlmostEqual(rel_error(dw, dw_num), 0., delta=1e-7, msg='dw for conv layer must agree with numeric gradient')
        self.assertAlmostEqual(rel_error(db, db_num), 0., delta=1e-7, msg='db for conv layer must agree with numeric gradient')


    def test_weighted_softmax_loss(self):

        batch_size  = 8
        num_classes = 5

        z = np.random.randn(batch_size, num_classes)
        y = np.random.randint(num_classes, size=batch_size)
        weights = np.random.randn(batch_size) # weights for loss function (not conventional weight matrix)

        # compute loss "manually"
        p = layers.softmax(z)
        logp = -np.log(p[range(batch_size), y])
        loss_ref = np.dot(weights, logp)

        # numeric gradients as reference
        dz_num = eval_numerical_gradient_array(lambda z: layers.weighted_softmax_loss(z, y, weights)[0], z, 1)

        loss, dz = layers.weighted_softmax_loss(z, y, weights)

        self.assertAlmostEqual(rel_error(loss, loss_ref), 0., delta=1e-14, msg='weighted softmax loss must agree with reference')
        self.assertAlmostEqual(rel_error(dz, dz_num), 0., delta=1e-7, msg='dz for weighted softmax loss must agree with numeric gradient')


def rel_error(x, y):
    """Compute relative errors."""
    return np.max(np.abs(x - y) / (np.maximum(np.abs(x) + np.abs(y), 1e-8)))


def eval_numerical_gradient_array(f, x, p, h=1e-5):
    """
    Approximate the numeric gradient of a function via
    the difference quotient (f(x + h) - f(x - h)) / (2 h),
    projected onto the direction p.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index

        xi_ref = x[i]
        x[i] = xi_ref + h
        fpos = f(x)         # evaluate f(x + h)
        x[i] = xi_ref - h
        fneg = f(x)         # evaluate f(x - h)
        x[i] = xi_ref       # restore

        # compute the partial derivative via centered difference quotient
        grad[i] = np.sum(p * (fpos - fneg)) / (2 * h)
        it.iternext()

    return grad


if __name__ == '__main__':
    unittest.main()
