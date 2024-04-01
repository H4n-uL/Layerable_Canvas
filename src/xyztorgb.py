import cv2, struct, zlib
import numpy as np
import matplotlib.pyplot as plt

layers = {}

with open('image.laca', 'rb') as f:
    header = f.read(64)
    json_len = struct.unpack('>I', header[4:8])[0]
    resolution = struct.unpack('>II', header[0x18:0x20])
    z_index = struct.unpack('>H', header[0xb:0xd])[0]
    if z_index not in layers:
        layers[z_index] = {
            'resolution': resolution,
            'data': [0]*resolution[0]
        }
    ljson = f.read(json_len)
    layers[z_index]

    while True:
        sign = f.read(4)
        if not sign: break
        if sign == b'\xff\xfd\x69\xc6':
            head = sign + f.read(28)
            lno = struct.unpack('>H', head[4:6])[0]
            col = struct.unpack('>I', head[0x8:0xc])[0]
            length = struct.unpack('>Q', head[0x10:0x18])[0]
            data = f.read(length)
            # data = zlib.decompress(f.read(length))
            layers[lno]['data'][col] = np.frombuffer(data, dtype='>f2').reshape((4, resolution[1])).T
        
    for l, i in zip(layers, range(len(layers))):
        data = np.array(layers[l]['data']).astype(np.float32)
        xyz, alpha = data[..., :3], data[..., 3]
        image = np.dstack((cv2.cvtColor(xyz.astype(np.float32), cv2.COLOR_XYZ2RGB), alpha))
        plt.imshow(image)
        plt.show()

        cv2.imwrite(f'srgbimg{i}.png', cv2.cvtColor(image*256, cv2.COLOR_RGBA2BGRA))
