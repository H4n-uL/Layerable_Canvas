import numpy as np
from tools import build_matrix

def RGBAtoLACA(data: np.ndarray, alpha: np.ndarray, profile):
    if profile is None: profile = open('sRGB.icc', 'rb').read()
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
        temp = np.transpose(data[0], (1, 0))[::-1]
        temp[0] = temp[0]**rTRC[0]
        temp[1] = temp[1]**gTRC[0]
        temp[2] = temp[2]**bTRC[0]
        retdata.append(np.vstack((RGB2XYZ.dot(temp), alpha[0])))
        data = data[1:]; alpha = alpha[1:]
    return np.array(retdata)