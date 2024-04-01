import cv2
from PIL import Image
import numpy as np
from tools import build

dt = {
    'uint8': 255,
    'uint16': 65535
}

images = ['Artwork KR CD-FCover-AdobeRGB.png']

open('image.laca', 'wb').write(b'')

for img, index in zip(images, range(len(images))):
    image = Image.open(img).convert('RGBA')
    alpha = np.array(image)[..., 3]
    roff = dt.get(str(alpha.dtype), 1)
    alpha = alpha.astype(np.float32) / roff

    data = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    roff = dt.get(str(data.dtype), 1)
    data = data.astype(np.float32) / roff
    resolution = data.shape[:2]

    xyz = cv2.cvtColor(data[..., :3], cv2.COLOR_BGR2XYZ)

    xyza = np.dstack((xyz, alpha))
    open('image.laca', 'ab').\
        write(build.layer('image', {}, 1.0, index, 'XYZ', 'json', (resolution[0], resolution[1]), (1, 1), ('0', '0'), '0', False, False))
    for rno in range(len(xyza)):
        row = np.transpose(xyza[rno], (1, 0)).astype('>f2').ravel().tobytes()
        # row = zlib.compress(row, level=9)
        open('image.laca', 'ab').write(build.horizimg_header(index, rno, row)+row)
