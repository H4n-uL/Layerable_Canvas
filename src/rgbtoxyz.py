import cv2, zlib
from PIL import Image, ImageCms
import numpy as np
from tools import build
import matplotlib.pyplot as plt
# import io
# import skimage

dt = {
    'uint8': 255,
    'uint16': 65535
}

images = ['ProPhoto.jpeg']

open('image.laca', 'wb').write(b'')

for img, index in zip(images, range(len(images))):
    image = Image.open(img).convert('RGBA')
    # profile = ImageCms.ImageCmsProfile(io.BytesIO(image.info.get('icc_profile')))
    # open('profile.icc', 'wb').write(image.info.get('icc_profile'))
    alpha = np.array(image)[..., 3]
    roff = dt.get(str(alpha.dtype), 1)
    alpha = alpha.astype(np.float32) / roff

    data = cv2.imread(img, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
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
