from __future__ import division, print_function

import numpy as np
import pytest
import tensorflow as tf

from conv2d import conv2d, conv2d_gradw, conv2d_gradx


@pytest.mark.parametrize("ii", range(5))
@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize("kernel_size", [(1, 1), (4, 5), (5, 4), (5, 5)])
@pytest.mark.parametrize("pad", ["SAME", "VALID"])
def test_conv2d(ii, stride, kernel_size, pad, rng):
    x = rng.rand(3, 5, 5, 2).astype("float32")
    w = rng.rand(*(kernel_size + (2, 1))).astype("float32")

    y = conv2d(x, w, pad=pad, stride=stride).ravel()
    xx = tf.constant(x, dtype="float32")
    ww = tf.constant(w, dtype="float32")
    yy = tf.nn.conv2d(xx, ww, strides=[1, stride[0], stride[1], 1], padding=pad)
    y_tf = yy.numpy()
    assert np.allclose(y, y_tf.ravel())


@pytest.mark.parametrize("ii", range(5))
@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize("kernel_size", [(1, 1), (4, 5), (5, 4), (5, 5)])
@pytest.mark.parametrize("pad", ["SAME", "VALID"])
def test_conv2d_gradw(ii, stride, kernel_size, pad, rng):
    x = rng.rand(3, 5, 5, 2).astype("float32")
    w = rng.rand(*(kernel_size + (2, 1))).astype("float32")

    # [N, H, W, K]
    y = conv2d(x, w, pad=pad, stride=stride)
    dy = np.random.rand(*y.shape)
    dw = conv2d_gradw(x, dy, ksize=w.shape[:2], pad=pad, stride=stride)

    # Tensorflow checks
    with tf.GradientTape() as tape:
        xx = tf.constant(x, dtype="float32")
        ww = tf.constant(w, dtype="float32")
        tape.watch(ww)
        dyy = tf.constant(dy, dtype="float32")
        yy = tf.nn.conv2d(xx, ww, strides=[1, stride[0], stride[1], 1], padding=pad)
        dww = tape.gradient(yy, ww, output_gradients=dyy)
    dw_tf = dww.numpy()
    assert np.allclose(dw.ravel(), dw_tf.ravel())


@pytest.mark.parametrize("ii", range(5))
@pytest.mark.parametrize("stride", [(1, 1), (2, 2), (3, 2)])
@pytest.mark.parametrize("kernel_size", [(1, 1), (4, 5), (5, 4), (5, 5)])
@pytest.mark.parametrize("pad", ["SAME", "VALID"])
def test_conv2d_gradx(ii, stride, kernel_size, pad, rng):
    x = rng.rand(3, 5, 5, 2).astype("float32")
    w = rng.rand(*(kernel_size + (2, 1))).astype("float32")

    # [N, H, W, K]
    y = conv2d(x, w, pad=pad, stride=stride)
    dy = np.random.rand(*y.shape)
    dx = conv2d_gradx(w, dy, x.shape[1:3], pad=pad, stride=stride)

    # Tensorflow checks
    with tf.GradientTape() as tape:
        xx = tf.constant(x, dtype="float32")
        tape.watch(xx)
        ww = tf.constant(w, dtype="float32")
        dyy = tf.constant(dy, dtype="float32")
        yy = tf.nn.conv2d(xx, ww, strides=[1, stride[0], stride[1], 1], padding=pad)
        dww = tape.gradient(yy, xx, output_gradients=dyy)
    dx_tf = dww.numpy()
    assert np.allclose(dx.ravel(), dx_tf.ravel())
