"""
Feedforward and backpropagation functions for various types of layers
composing an artificial neural network.

Code inspired by
    Stanford CS231n Convolutional Neural Networks for Visual Recognition
    http://cs231n.stanford.edu, http://cs231n.github.io
"""

import numpy as np


def affine_forward(x, w, b):
    """
    Compute the forward pass for an affine layer `z = w x + b`.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Args:
        x: numpy array containing input data, of shape (N, d_1, ..., d_k)
        w: numpy array of weights, of shape (M, D)
        b: numpy array of biases, of shape (M,)

    Returns a tuple of:
        out: output, of shape (N, M)
        cache: (x, w, b)
    """
    # reshape into dimension (N, D)
    xp = np.reshape(x, (x.shape[0], -1))
    # multiply with transposed weight matrix from right to
    # keep minibatch dimension as first dimension
    out = np.dot(xp, w.T) + b
    cache = (x, w, b)
    return (out, cache)


def affine_backpropagate(dout, cache):
    """
    Compute the backward pass for an affine layer `z = w x + b`.

    Args:
        dout: upstream derivative, of shape (N, M)
        cache: cache variable filled during forward pass

    Returns a tuple of:
        dx: gradient with respect to x, of shape (N, d1, ..., d_k)
        dw: gradient with respect to w, of shape (M, D)
        db: gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    # backpropagate gradient to input of current layer
    dx = np.reshape(np.dot(dout, w), x.shape)
    # reshape x into dimension (N, D)
    xp = np.reshape(x, (x.shape[0], -1))
    # backpropagate gradient to weights and biases, effectively summing over N
    dw = np.dot(dout.T, xp)
    db = np.sum(dout, axis=0)
    return (dx, dw, db)


def sigmoid_forward(x):
    """
    Compute the forward pass for a layer of sigmoid units.

    Args:
        x: inputs, of any shape

    Returns a tuple of:
        out: output, of the same shape as x
        cache: (x, out)
    """
    # numerically stable implementation of 1 / (1 + exp(-x))
    pos_mask = (x >= 0)
    neg_mask = (x <  0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp( x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    out = top / (1 + z)
    cache = (x, out)
    return (out, cache)


def sigmoid_backpropagate(dout, cache):
    """
    Compute the backward pass for a layer of sigmoid units.

    Args:
        dout: upstream derivatives, of any shape
        cache: cache variable filled during forward pass

    Returns:
        dx: gradient with respect to x
    """
    (x, out) = cache
    dx = out*(1 - out) * dout
    return dx


def relu_forward(x):
    """
    Compute the forward pass for a layer of rectified linear units (ReLUs).

    Args:
        x: inputs, of any shape

    Returns a tuple of:
        out: output, of the same shape as x
        cache: x
    """
    out = np.maximum(x, 0)
    cache = x
    return (out, cache)


def relu_backpropagate(dout, cache):
    """
    Compute the backward pass for a layer of rectified linear units (ReLUs).

    Args:
        dout: upstream derivatives, of same shape as input x in forward pass
        cache: cache variable filled during forward pass

    Returns:
        dx: Gradient with respect to x
    """
    x = cache
    dx = (x >= 0).astype(x.dtype) * dout
    return dx



def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine (fully connected) transform followed by a ReLU layer.

    Args:
        x: input to the affine layer
        w, b: weights for the affine layer

    Returns a tuple of:
        out: output from the ReLU layer
        cache: object to give to the backward pass
    """
    a,   fc_cache = affine_forward(x, w, b)
    out, sg_cache = relu_forward(a)
    cache = (fc_cache, sg_cache)
    return (out, cache)


def affine_relu_backpropagate(dout, cache):
    """
    Backward pass for the affine-ReLU convenience layer.
    """
    fc_cache, sg_cache = cache
    da         = relu_backpropagate(dout, sg_cache)
    dx, dw, db = affine_backpropagate(da, fc_cache)
    return (dx, dw, db)


def conv_forward(x, w, b, conv_param):
    """
    Fast implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Args:
        x: input data of shape (N, C, H, W)
        w: filter weights of shape (F, C, HH, WW)
        b: biases, of shape (F,)
        conv_param: dictionary with the following keys:
            'stride': number of pixels between adjacent receptive fields in the
                      horizontal and vertical directions
            'pad': number of pixels that will be used to symmetrically zero-pad the input

    Returns a tuple of:
        out: output data, of shape (N, F, H', W') where H' and W' are given by
             H' = 1 + (H + 2*pad - HH) / stride
             W' = 1 + (W + 2*pad - WW) / stride
        cache: (x, w, b, conv_param)
    """
    (N, C, H, W)   = x.shape
    (F, C, HH, WW) = w.shape
    pad    = conv_param['pad']
    stride = conv_param['stride']

    # pad the input
    xpad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    H += 2 * pad
    W += 2 * pad

    # output dimensions
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1

    # perform an im2col operation by picking clever strides
    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    # take number of bytes of each entry into account
    strides = x.itemsize * np.array(strides)
    x_stride = np.lib.stride_tricks.as_strided(xpad, shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)

    # now all our convolutions are a big matrix multiply
    res = np.dot(w.reshape((F, -1)), x_cols) + b.reshape((F, 1))

    # reshape the output
    res.shape = (F, N, out_h, out_w)
    out = res.transpose((1, 0, 2, 3))

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backpropagate(dout, cache):
    """
    Fast implementation of the backward pass for a convolutional layer.

    Args:
        dout: upstream derivatives of the cost function
        cache: tuple of (x, w, b, conv_param) as in conv_forward

    Returns a tuple of:
        dx: gradient with respect to x
        dw: gradient with respect to w
        db: gradient with respect to b
    """
    x, w, b, conv_param, x_cols = cache

    (N, C, H, W)   = x.shape
    (F, C, HH, WW) = w.shape
    pad    = conv_param['pad']
    stride = conv_param['stride']

    # flip entries of weight filter along spatial dimensions
    wflip = np.flip(np.flip(w, 2), 3)
    # transpose feature and channel dimensions
    wflip = wflip.transpose((1, 0, 2, 3))

    # interleave zeros to mimic effect of stride
    doutp = np.zeros((dout.shape[0], dout.shape[1], stride*dout.shape[2], stride*dout.shape[3]))
    doutp[:, :, ::stride, ::stride] = dout
    # pad pixels with zeros on each side along the height and width axes
    # to take shifted indices of weight tensor into account
    doutp = np.pad(doutp, ((0, 0), (0, 0), (HH-1, HH-1), (WW-1, WW-1)), mode='constant')
    doutp = doutp[:, :, pad:, pad:]
    doutp = np.ascontiguousarray(doutp)

    # perform an im2col operation on dout by picking clever strides
    _, _, dp_h, dp_w = doutp.shape
    shape = (F, HH, WW, N, H, W)
    strides = (dp_h * dp_w, dp_w, 1, F * dp_h * dp_w, dp_w, 1)
    # take number of bytes of each entry into account
    strides = doutp.itemsize * np.array(strides)
    dout_cols = np.lib.stride_tricks.as_strided(doutp, shape=shape, strides=strides)
    dout_cols = np.ascontiguousarray(dout_cols)
    dout_cols.shape = (F * HH * WW, N * H * W)

    # compute convolutions by a large matrix multiplication
    dx = np.dot(wflip.reshape((C, -1)), dout_cols)

    # reshape the gradient and move minibatch dimension N to the front
    dx.shape = (C, N, H, W)
    dx = dx.transpose((1, 0, 2, 3))

    dout_reshaped = dout.transpose((1, 0, 2, 3)).reshape((F, -1))
    # effectively sum over N
    dw = np.dot(dout_reshaped, x_cols.T).reshape(w.shape)

    db = np.sum(dout, axis=(0, 2, 3))

    return (dx, dw, db)


def softmax(z):
    """
    Compute the softmax function p_j = exp(z_j) / sum_k exp(z_k).
    """
    # subtract maximum to avoid overflow
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    p /= np.sum(p, axis=1, keepdims=True)
    return p


def weighted_softmax_loss(z, y, weights):
    """
    Compute the weighted loss and gradient for softmax classification.

    Args:
        z: network output, of shape (N, C) where z[i, j] is the score
           for the jth class for the ith minibatch element
        y: vector of labels, of shape (N,) where y[i] is the label for z[i] and
           0 <= y[i] < C
        weights: the loss corresponding to the ith minibatch element
                 is scaled by weights[i]

    Returns a tuple of:
        loss: softmax cost "-log(a_y)" with a_j = exp(z_j) / Z
        dz: gradient of the loss with respect to z
    """
    # subtract maximum to avoid overflow
    z = z - np.max(z, axis=1, keepdims=True)
    # normalization factor S = sum_j exp(z_j)
    S = np.sum(np.exp(z), axis=1, keepdims=True)
    # log(exp(z_j) / S) = z_j - log(S)
    log_probs = z - np.log(S)
    # minibatch size
    N = z.shape[0]
    # weighted "-log(a_y)" with a_j = exp(z_j) / S and y the label index
    loss = np.dot(weights, -log_probs[np.arange(N), y])
    # a_j - y_j
    dz = np.exp(log_probs)
    dz[np.arange(N), y] -= 1
    dz *= weights[:, None]

    return loss, dz
