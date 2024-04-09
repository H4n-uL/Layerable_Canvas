import numpy as np

def scan(coords, shape):
    ndim = len(coords)
    i = 0
    while True:
        yield tuple(coords), i
        for j in range(ndim - 1, -1, -1):
            if coords[j] < shape[j] - 1:
                coords[j] += 1; break
            elif j > 0: coords[j] = 0
            else: return
        i += 1

def zigzag(x):
    shape = x.shape
    ndim = x.ndim
    y = np.zeros(np.prod(shape)).astype(x.dtype)
    coords = [0] * ndim
    for idx, i in scan(coords, shape):
        if sum(idx) % 2 == 0: y[i] = x[idx]
        else: y[i] = x[tuple(idx[j] for j in range(ndim - 1, -1, -1))]
    return y

def inverse_zigzag(y, shape):
    ndim = len(shape)
    x = np.zeros(shape).astype(y.dtype)
    coords = [0] * ndim
    for idx, i in scan(coords, shape):
        if sum(idx) % 2 == 0: x[idx] = y[i]
        else: x[tuple(idx[j] for j in range(ndim - 1, -1, -1))] = y[i]
    return x

input_tensor = np.arange(64).reshape((8, 8))
output = zigzag(input_tensor)
print(output)

shape = (8, 8)
reconstructed_tensor = inverse_zigzag(output, shape)
print(reconstructed_tensor)
