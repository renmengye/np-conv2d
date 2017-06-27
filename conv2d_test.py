from __future__ import division, print_function

import numpy as np
import tensorflow as tf

from conv2d import conv2d, calc_pad, conv2d_gradw, conv2d_gradx


def test(x, w, pad='SAME', stride=(1, 1)):
    y = conv2d(x, w, pad=pad, stride=stride).ravel()
    xx = tf.constant(x, dtype='float32')
    ww = tf.constant(w, dtype='float32')
    yy = tf.nn.conv2d(
        xx, ww, strides=[1, stride[0], stride[1], 1], padding=pad)
    with tf.Session() as sess:
        y_tf = yy.eval().ravel()
    np.testing.assert_almost_equal(y, y_tf, decimal=3)


def test_gradw(x, w, pad='SAME', stride=(1, 1)):
    # [N, H, W, K]
    y = conv2d(x, w, pad=pad, stride=stride)
    dy = np.random.rand(*y.shape)
    dw = conv2d_gradw(x, dy, ksize=w.shape[:2], pad=pad, stride=stride)

    # Tensorflow checks
    xx = tf.constant(x, dtype='float32')
    ww = tf.constant(w, dtype='float32')
    dyy = tf.constant(dy, dtype='float32')
    yy = tf.nn.conv2d(
        xx, ww, strides=[1, stride[0], stride[1], 1], padding=pad)
    dww = tf.squeeze(tf.gradients(yy, ww, dyy), [0])
    with tf.Session() as sess:
        dw_tf = dww.eval()
    np.testing.assert_almost_equal(dw.ravel(), dw_tf.ravel(), decimal=3)


def test_gradx(x, w, pad='SAME', stride=(1, 1)):
    # [N, H, W, K]
    y = conv2d(x, w, pad=pad, stride=stride)
    dy = np.random.rand(*y.shape)
    dx = conv2d_gradx(w, dy, x.shape[1:3], pad=pad, stride=stride)

    # Tensorflow checks
    xx = tf.constant(x, dtype='float32')
    ww = tf.constant(w, dtype='float32')
    dyy = tf.constant(dy, dtype='float32')
    yy = tf.nn.conv2d(
        xx, ww, strides=[1, stride[0], stride[1], 1], padding=pad)
    dww = tf.squeeze(tf.gradients(yy, xx, dyy), [0])
    with tf.Session() as sess:
        dx_tf = dww.eval()
    np.testing.assert_almost_equal(dx.ravel(), dx_tf.ravel(), decimal=3)


if __name__ == '__main__':
    np.random.seed(0)

    for ii in range(25):
        x = np.random.rand(3, 5, 5, 2).astype('float32')
        w = np.random.rand(2, 3, 2, 1).astype('float32')
        test(x, w)
        test_gradw(x, w)
        test_gradx(x, w)
        print(ii, 'pass')

    for ii in range(25):
        x = np.random.rand(3, 5, 5, 2).astype('float32')
        w = np.random.rand(2, 3, 2, 1).astype('float32')
        test(x, w, pad='VALID')
        test_gradw(x, w, pad='VALID')
        test_gradx(x, w, pad='VALID')
        print(ii, 'pass')

    for ii in range(25):
        x = np.random.rand(3, 5, 5, 2).astype('float32')
        w = np.random.rand(2, 3, 2, 1).astype('float32')
        test(x, w, stride=(2, 2))
        test_gradw(x, w, stride=(2, 2))
        test_gradx(x, w, stride=(2, 2))
        print(ii, 'pass')

    for ii in range(25):
        x = np.random.rand(3, 5, 5, 2).astype('float32')
        w = np.random.rand(3, 3, 2, 1).astype('float32')
        test(x, w, pad='VALID', stride=(2, 2))
        test_gradw(x, w, pad='VALID', stride=(2, 2))
        test_gradx(x, w, pad='VALID', stride=(2, 2))
        print(ii, 'pass')