import cv2
from PIL import Image
import numpy as np
from tools import build

images = ['image.png']

open('image.laca', 'wb').write(b'')

for img, index in zip(images, range(len(images))):
    image = Image.open(img)
    data = np.array(image.convert('RGBA')).astype(np.float32) / 255.0
    resolution = data.shape[:2]

    xyz = cv2.cvtColor(data[..., :3], cv2.COLOR_RGB2XYZ)
    alpha = data[..., 3]

    xyza = np.dstack((xyz, alpha))
    data = b''
    for rno in range(len(xyza)):
        row = np.transpose(xyza[rno], (1, 0)).astype('>f2').ravel().tobytes()
        header = build.horizimg_header(index, rno, row)
        data += header + row

    x = build.layer('image', {}, 1.0, index, 'XYZ', 'json', (resolution[0], resolution[1]), (1, 1), ('0', '0'), '0', False, False)

    open('image.laca', 'ab').write(x + data)
