from __future__ import division

import numpy as np


def array_offset(x):
    """Get offset of array data from base data in bytes."""
    if x.base is None:
        return 0

    base_start = x.base.__array_interface__['data'][0]
    start = x.__array_interface__['data'][0]
    return start - base_start


def calc_pad(pad, in_siz, out_siz, stride, ksize):
    """Calculate padding width.

    Args:
        pad: padding method, "SAME", "VALID", or manually speicified.
        ksize: kernel size [I, J].

    Returns:
        pad_: Actual padding width.
    """
    if pad == 'SAME':
        return max((out_siz - 1) * stride + ksize - in_siz, 0)
    elif pad == 'VALID':
        return 0
    else:
        return pad


def calc_gradx_pad(pad, in_siz, out_siz, stride, ksize):
    """Calculate padding width for conv2d_gradx.

    Args:
        pad: Padding method, "SAME", "VALID", or manually speicified.
        in_siz: Size of the input to `conv2d_gradx` (i.e. size of `dy`).
        out_siz: Size of the output of `conv2d_gradx` (i.e. size of `dx`).
        stride: Length of the convolution stride.
        ksize: Kernel size.

    Returns:
        pad_: Actual padding width.
    """
    if pad == 'SAME':
        out_siz_min = (in_siz - 1) * stride + 1
        p = out_siz + ksize - 1 - out_siz_min
        p = max(p, 0)
        p = min(p, (ksize - 1) * 2)
        return p
    elif pad == 'VALID':
        return (ksize - 1) * 2
    else:
        return pad


def calc_size(h, kh, pad, sh):
    """Calculate output image size on one dimension.

    Args:
        h: input image size.
        kh: kernel size.
        pad: padding strategy.
        sh: stride.

    Returns:
        s: output size.
    """

    if pad == 'VALID':
        return np.ceil((h - kh + 1) / sh)
    elif pad == 'SAME':
        return np.ceil(h / sh)
    else:
        return int(np.ceil((h - kh + pad + 1) / sh))


def extract_sliding_windows_gradw(x,
                                  ksize,
                                  pad,
                                  stride,
                                  orig_size,
                                  floor_first=True):
    """Extracts dilated windows.

    Args:
        x: [N, H, W, C]
        k: [KH, KW]
        pad: [PH, PW]
        stride: [SH, SW]

    Returns:
        y: [N, H', W', KH, KW, C]
    """
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]
    kh = ksize[0]
    kw = ksize[1]
    sh = stride[0]
    sw = stride[1]

    h2 = orig_size[0]
    w2 = orig_size[1]
    ph = int(calc_pad(pad, h, h2, 1, ((kh - 1) * sh + 1)))
    pw = int(calc_pad(pad, w, w2, 1, ((kw - 1) * sw + 1)))

    ph2 = int(np.ceil(ph / 2))
    ph3 = int(np.floor(ph / 2))
    pw2 = int(np.ceil(pw / 2))
    pw3 = int(np.floor(pw / 2))
    if floor_first:
        pph = (ph3, ph2)
        ppw = (pw3, pw2)
    else:
        pph = (ph2, ph3)
        ppw = (pw2, pw3)
    x = np.pad(
        x, ((0, 0), (ph3, ph2), (pw3, pw2), (0, 0)),
        mode='constant',
        constant_values=(0.0, ))
    p2h = (-x.shape[1]) % sh
    p2w = (-x.shape[2]) % sw
    if p2h > 0 or p2w > 0:
        x = np.pad(
            x, ((0, 0), (0, p2h), (0, p2w), (0, 0)),
            mode='constant',
            constant_values=(0.0, ))

    # The following code extracts window without copying the data:
    # x = x.reshape([n, int(x.shape[1] / sh), sh, int(x.shape[2] / sw), sw, c])
    # y = np.zeros([n, h2, w2, kh, kw, c])
    # for ii in range(h2):
    #     for jj in range(w2):
    #         h0 = int(np.floor(ii / sh))
    #         w0 = int(np.floor(jj / sw))
    #         y[:, ii, jj, :, :, :] = x[:, h0:h0 + kh, ii % sh, w0:w0 + kw, jj %
    #                                   sw, :]
    x_sn, x_sh, x_sw, x_sc = x.strides
    y_strides = (x_sn, x_sh, x_sw, sh * x_sh, sw * x_sw, x_sc)
    y = np.ndarray((n, h2, w2, kh, kw, c),
                   dtype=x.dtype,
                   buffer=x.data,
                   offset=array_offset(x),
                   strides=y_strides)
    return y


def extract_sliding_windows_gradx(x,
                                  ksize,
                                  pad,
                                  stride,
                                  orig_size,
                                  floor_first=False):
    """Extracts windows on a dilated image.

    Args:
        x: [N, H', W', C] (usually dy)
        k: [KH, KW]
        pad: [PH, PW]
        stride: [SH, SW]
        orig_size: [H, W]

    Returns:
        y: [N, H, W, KH, KW, C]
    """
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]
    kh = ksize[0]
    kw = ksize[1]
    ph = pad[0]
    pw = pad[1]
    sh = stride[0]
    sw = stride[1]
    h2 = orig_size[0]
    w2 = orig_size[1]
    xs = np.zeros([n, x.shape[1], sh, x.shape[2], sw, c])
    xs[:, :, 0, :, 0, :] = x
    xss = xs.shape
    x = xs.reshape([xss[0], xss[1] * xss[2], xss[3] * xss[4], xss[5]])
    x = x[:, :h2, :w2, :]

    ph2 = int(np.ceil(ph / 2))
    ph3 = int(np.floor(ph / 2))
    pw2 = int(np.ceil(pw / 2))
    pw3 = int(np.floor(pw / 2))
    if floor_first:
        pph = (ph3, ph2)
        ppw = (pw3, pw2)
    else:
        pph = (ph2, ph3)
        ppw = (pw2, pw3)
    x = np.pad(
        x, ((0, 0), pph, ppw, (0, 0)),
        mode='constant',
        constant_values=(0.0, ))

    # The following code extracts window without copying the data:
    # y = np.zeros([n, h2, w2, kh, kw, c])
    # for ii in range(h2):
    #     for jj in range(w2):
    #         y[:, ii, jj, :, :, :] = x[:, ii:ii + kh, jj:jj + kw, :]
    x_sn, x_sh, x_sw, x_sc = x.strides
    y_strides = (x_sn, x_sh, x_sw, x_sh, x_sw, x_sc)
    y = np.ndarray((n, h2, w2, kh, kw, c),
                   dtype=x.dtype,
                   buffer=x.data,
                   offset=array_offset(x),
                   strides=y_strides)
    return y


def extract_sliding_windows(x, ksize, pad, stride, floor_first=True):
    """Converts a tensor to sliding windows.

    Args:
        x: [N, H, W, C]
        k: [KH, KW]
        pad: [PH, PW]
        stride: [SH, SW]

    Returns:
        y: [N, (H-KH+PH+1)/SH, (W-KW+PW+1)/SW, KH * KW, C]
    """
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]
    kh = ksize[0]
    kw = ksize[1]
    sh = stride[0]
    sw = stride[1]

    h2 = int(calc_size(h, kh, pad, sh))
    w2 = int(calc_size(w, kw, pad, sw))
    ph = int(calc_pad(pad, h, h2, sh, kh))
    pw = int(calc_pad(pad, w, w2, sw, kw))

    ph0 = int(np.floor(ph / 2))
    ph1 = int(np.ceil(ph / 2))
    pw0 = int(np.floor(pw / 2))
    pw1 = int(np.ceil(pw / 2))

    if floor_first:
        pph = (ph0, ph1)
        ppw = (pw0, pw1)
    else:
        pph = (ph1, ph0)
        ppw = (pw1, pw0)
    x = np.pad(
        x, ((0, 0), pph, ppw, (0, 0)),
        mode='constant',
        constant_values=(0.0, ))

    # The following code extracts window without copying the data:
    # y = np.zeros([n, h2, w2, kh, kw, c])
    # for ii in range(h2):
    #     for jj in range(w2):
    #         xx = ii * sh
    #         yy = jj * sw
    #         y[:, ii, jj, :, :, :] = x[:, xx:xx + kh, yy:yy + kw, :]
    x_sn, x_sh, x_sw, x_sc = x.strides
    y_strides = (x_sn, sh * x_sh, sw * x_sw, x_sh, x_sw, x_sc)
    y = np.ndarray((n, h2, w2, kh, kw, c),
                   dtype=x.dtype,
                   buffer=x.data,
                   offset=array_offset(x),
                   strides=y_strides)
    return y


def conv2d(x, w, pad='SAME', stride=(1, 1)):
    """2D convolution (technically speaking, correlation).

    Args:
        x: [N, H, W, C]
        w: [I, J, C, K]
        pad: [PH, PW]
        stride: [SH, SW]

    Returns:
        y: [N, H', W', K]
    """
    ksize = w.shape[:2]
    x = extract_sliding_windows(x, ksize, pad, stride)
    ws = w.shape
    w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
    xs = x.shape
    x = x.reshape([xs[0] * xs[1] * xs[2], -1])
    y = x.dot(w)
    y = y.reshape([xs[0], xs[1], xs[2], -1])
    return y


def conv2d_gradw(x, dy, ksize, pad='SAME', stride=(1, 1)):
    """2D convolution gradient wrt. filters.

    Args:
        dy: [N, H', W', K]
        x: [N, H, W, C]
        ksize: original w ksize [I, J].

    Returns:
        dw: [I, J, C, K]
    """
    dy = np.transpose(dy, [1, 2, 0, 3])
    x = np.transpose(x, [3, 1, 2, 0])
    ksize2 = dy.shape[:2]
    x = extract_sliding_windows_gradw(x, ksize2, pad, stride, ksize)
    dys = dy.shape
    dy = dy.reshape([dys[0] * dys[1] * dys[2], dys[3]])
    xs = x.shape
    x = x.reshape([xs[0] * xs[1] * xs[2], -1])
    dw = x.dot(dy)
    dw = dw.reshape([xs[0], xs[1], xs[2], -1])
    dw = np.transpose(dw, [1, 2, 0, 3])
    dw = dw[:ksize[0], :ksize[1], :, :]
    return dw


def conv2d_gradx(w, dy, xsize, pad='SAME', stride=(1, 1)):
    """2D convolution gradient wrt. input.

    Args:
        dy: [N, H', W', K]
        w: [I, J, C, K]
        xsize: Original image size, [H, W]

    Returns:
        dx: [N, H, W, C]
    """
    assert w.shape[-1] == dy.shape[-1], "`w` filters must match `dy` channels"
    w = np.transpose(w, [0, 1, 3, 2])

    dys = dy.shape[1:3]
    ksize = w.shape[:2]
    pad2 = (
        calc_gradx_pad(pad, dys[0], xsize[0], stride[0], ksize[0]),
        calc_gradx_pad(pad, dys[1], xsize[1], stride[1], ksize[1]),
    )

    dx = extract_sliding_windows_gradx(dy, ksize, pad2, stride, xsize)
    dxs = dx.shape
    dx = dx.reshape([dxs[0] * dxs[1] * dxs[2], -1])
    w = w[::-1, ::-1, :, :]
    ws = w.shape
    w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
    dx = dx.dot(w)
    return dx.reshape([dxs[0], dxs[1], dxs[2], -1])
