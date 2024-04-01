import cv2, zlib
from PIL import Image, ImageCms
import numpy as np
from tools import build
from io import BytesIO

images = ['Artwork KR CD-FCover-AdobeRGB.png']

open('image.laca', 'wb').write(b'')

for img, index in zip(images, range(len(images))):
    image = Image.open(img).convert('RGBA')
    src_profile = image.info.get('icc_profile', '')
    if src_profile:
        src_profile = ImageCms.ImageCmsProfile(BytesIO(src_profile))
    else:
        src_profile = ImageCms.createProfile('sRGB')

    image = ImageCms.profileToProfile(image, src_profile, ImageCms.createProfile('sRGB'), outputMode='RGBA')
    data = np.asarray(image, dtype=np.float32)/255
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
