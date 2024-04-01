import cv2, zlib
from PIL import Image, ImageCms
import numpy as np
from tools import build
from io import BytesIO

images = ['image.png']

open('image.laca', 'wb').write(b'')

for img, index in zip(images, range(len(images))):
    image = Image.open(img)
    # prf = image.info.get('icc_profile')
    # srcprf = prf and ImageCms.ImageCmsProfile(BytesIO(prf)) or ImageCms.createProfile('sRGB')
    # destprf = ImageCms.createProfile('XYZ')
    # rgbxyz = ImageCms.buildTransformFromOpenProfiles(srcprf, destprf, 'RGB', 'XYZ')
    # ImageCms.applyTransform(image, rgbxyz)

    data = np.array(image.convert('RGBA')).astype(np.float32) / 256.0
    resolution = data.shape[:2]

    xyz = cv2.cvtColor(data[..., :3], cv2.COLOR_RGB2XYZ)
    alpha = data[..., 3]

    xyza = np.dstack((xyz, alpha))
    open('image.laca', 'ab').\
        write(build.layer('image', {}, 1.0, index, 'XYZ', 'json', (resolution[0], resolution[1]), (1, 1), ('0', '0'), '0', False, False))
    for rno in range(len(xyza)):
        row = np.transpose(xyza[rno], (1, 0)).astype('>f2').ravel().tobytes()
        # row = zlib.compress(row, level=9)
        open('image.laca', 'ab').write(build.horizimg_header(index, rno, row)+row)
