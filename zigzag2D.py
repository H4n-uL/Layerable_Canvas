import numpy as np

def scan(v, h, i, vmax, hmax):
    i += 1
    if (h + v) % 2 == 0:
        if v == 0 and h < hmax - 1: h += 1
        elif h == hmax - 1: v += 1
        else: v -= 1; h += 1
    else:
        if v == vmax - 1: h += 1
        elif h == 0: v += 1
        else: v += 1; h -= 1
    return v, h, i

def zigzag(x):
    v, h, i = 0, 0, 0
    vmax, hmax = x.shape
    y = np.zeros(vmax * hmax).astype(x.dtype)
    while v < vmax and h < hmax:
        y[i] = x[v, h]
        v, h, i = scan(v, h, i, vmax, hmax)
    return y[:i]

def inverse_zigzag(y, vmax, hmax):
    v, h, i = 0, 0, 0
    x = np.zeros((vmax, hmax)).astype(y.dtype)
    while v < vmax and h < hmax:
        x[v, h] = y[i]
        v, h, i = scan(v, h, i, vmax, hmax)
    return x

input_tensor = np.arange(64).reshape((8, 8))
output = zigzag(input_tensor)
print(output)

reconstructed_tensor = inverse_zigzag(output, 8, 8)
print(reconstructed_tensor)