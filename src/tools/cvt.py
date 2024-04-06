import numpy as np
from tools import build_matrix
import os

sRGB_Profile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sRGB.icc')

# import struct
# def getxyz(data: bytes):
#     x = struct.unpack('>h', data[:2])[0] + struct.unpack('>H', data[2:4])[0] / 2**16
#     y = struct.unpack('>h', data[4:6])[0] + struct.unpack('>H', data[6:8])[0] / 2**16
#     z = struct.unpack('>h', data[8:10])[0] + struct.unpack('>H', data[10:])[0] / 2**16
#     return x, y, z

# rXYZ = b'\x00\x01\x0c\x44\x00\x00\x07\x94\xff\xff\xfd\xa2'
# gXYZ = b'\x00\x00\x05\xdf\x00\x00\xfd\x8f\x00\x00\x03\xdb'
# bXYZ = b'\xff\xff\xf3\x26\xff\xff\xfb\xa1\x00\x00\xc0\x75'

# D65TOD50 = np.array([getxyz(rXYZ), getxyz(gXYZ), getxyz(bXYZ)]).T
D65TOD50 = np.array([
[1.04791259765625, 0.0229339599609375, -0.050201416015625],
[0.02960205078125, 0.9904632568359375, -0.0170745849609375],
[-0.009246826171875, 0.0150604248046875,0.7517852783203125]])
D50TOD65 = np.linalg.inv(D65TOD50)

def RGBAtoLACA(data: np.ndarray, alpha: np.ndarray, profile):
    if profile is None: profile = open(sRGB_Profile, 'rb').read()
    rXYZ = build_matrix.get_cXYZ(profile, b'r')
    gXYZ = build_matrix.get_cXYZ(profile, b'g')
    bXYZ = build_matrix.get_cXYZ(profile, b'b')

    rTRC = build_matrix.get_cTRC(profile, b'r')
    gTRC = build_matrix.get_cTRC(profile, b'g')
    bTRC = build_matrix.get_cTRC(profile, b'b')

    RGB2XYZ = np.array([rXYZ, gXYZ, bXYZ]).T

    retdata = []
    while data.size > 0:
        # BGR to RGB
        row = np.transpose(data[0], (1, 0))[::-1]
        row[0] = rTRC['toXYZ'](row[0])
        row[1] = gTRC['toXYZ'](row[1])
        row[2] = bTRC['toXYZ'](row[2])
        row = RGB2XYZ.dot(row)
        row = D50TOD65.dot(row)
        retdata.append(np.vstack((row, alpha[0])))
        data = data[1:]; alpha = alpha[1:]
    return np.array(retdata)

def LACAtoRGBA(data: np.ndarray, profile):
    if profile is None: profile = open(sRGB_Profile, 'rb').read()
    rXYZ = build_matrix.get_cXYZ(profile, b'r')
    gXYZ = build_matrix.get_cXYZ(profile, b'g')
    bXYZ = build_matrix.get_cXYZ(profile, b'b')

    rTRC = build_matrix.get_cTRC(profile, b'r')
    gTRC = build_matrix.get_cTRC(profile, b'g')
    bTRC = build_matrix.get_cTRC(profile, b'b')

    XYZ2RGB = np.linalg.inv(np.array([rXYZ, gXYZ, bXYZ]).T)

    retdata = []
    while data.size > 0:
        # BGR to RGB
        alpha = data[0][3]
        row = D65TOD50.dot(data[0][:3]) # D65 XYZ to D50 XYZ
        row = XYZ2RGB.dot(row)          # XYZ to RGB
        # Gamma
        row[0] = rTRC['toRGB'](row[0])
        row[1] = gTRC['toRGB'](row[1])
        row[2] = bTRC['toRGB'](row[2])
        row = row[::-1] # RGB to BGR
        row = np.transpose(np.vstack((row, alpha)), (1, 0))
        retdata.append(row)
        data = data[1:]; alpha = alpha[1:]
    return np.array(retdata)
