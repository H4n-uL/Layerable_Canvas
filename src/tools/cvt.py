import numpy as np
from tools import build_matrix
import os

sRGB_Profile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sRGB.icc')

D50TOD65 = np.array([
[ 0.9554734527042182,  -0.023098536874261423, 0.0632593086610217],
[-0.028369706963208136, 1.0099954580058652,   0.021041398966943008],
[0.012314001688319899, -0.020507696433477912, 1.3303659366080753]])
D65TOD50 = np.linalg.inv(D50TOD65)

def RGBAtoLACA(data: np.ndarray, alpha: np.ndarray, profile):
    if profile is None: profile = open(sRGB_Profile, 'rb').read()
    rXYZ = build_matrix.get_cXYZ(profile, b'r')
    gXYZ = build_matrix.get_cXYZ(profile, b'g')
    bXYZ = build_matrix.get_cXYZ(profile, b'b')

    rTRC = build_matrix.get_cTRC(profile, b'r')
    gTRC = build_matrix.get_cTRC(profile, b'g')
    bTRC = build_matrix.get_cTRC(profile, b'b')
    if rTRC[1] != 'gamma': rTRC = build_matrix.table2gamma(rTRC[0])
    if gTRC[1] != 'gamma': gTRC = build_matrix.table2gamma(gTRC[0])
    if bTRC[1] != 'gamma': bTRC = build_matrix.table2gamma(bTRC[0])

    RGB2XYZ = np.array([rXYZ, gXYZ, bXYZ]).T

    retdata = []
    while data.size > 0:
        # BGR to RGB
        row = np.transpose(data[0], (1, 0))[::-1]
        row[0] = row[0]**rTRC[0]
        row[1] = row[1]**gTRC[0]
        row[2] = row[2]**bTRC[0]
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
    if rTRC[1] != 'gamma': rTRC = build_matrix.table2gamma(rTRC[0])
    if gTRC[1] != 'gamma': gTRC = build_matrix.table2gamma(gTRC[0])
    if bTRC[1] != 'gamma': bTRC = build_matrix.table2gamma(bTRC[0])

    XYZ2RGB = np.linalg.inv(np.array([rXYZ, gXYZ, bXYZ]).T)

    retdata = []
    while data.size > 0:
        # BGR to RGB
        alpha = data[0][3]
        row = D65TOD50.dot(data[0][:3]) # D65 XYZ to D50 XYZ
        row = XYZ2RGB.dot(row)          # XYZ to RGB
        # Gamma
        row[0] = row[0]**(1/rTRC[0])
        row[1] = row[1]**(1/gTRC[0])
        row[2] = row[2]**(1/bTRC[0])
        row = row[::-1] # RGB to BGR
        row = np.transpose(np.vstack((row, alpha)), (1, 0))
        retdata.append(row)
        data = data[1:]; alpha = alpha[1:]
    return np.array(retdata)
