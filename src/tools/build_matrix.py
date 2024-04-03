import struct
import numpy as np
from PIL import Image, ImageCms
from scipy.optimize import curve_fit

# img = 'ProPhoto.jpeg'
img = 'image.png'
profile = Image.open(img).info.get('icc_profile')
if profile is None: profile = open('sRGB.icc', 'rb').read()


def parse_s15Fixed16Number(data: bytes):
    '''s15Fixed16Number
    16bit signed integer + 16bit unsigned integer as fraction
    0x8000 0000 = -32768 + 0/2^16
    0x0000 0000 = 0 + 0/2^16 = 0
    0x0001 0000 = 1 + 0/2^16 = 1.0
    0x7fff ffff = 32767 + 65535/2^16
    '''
    x = struct.unpack('>h', data[:2])[0] + struct.unpack('>H', data[2:4])[0] / 2**16
    y = struct.unpack('>h', data[4:6])[0] + struct.unpack('>H', data[6:8])[0] / 2**16
    z = struct.unpack('>h', data[8:10])[0] + struct.unpack('>H', data[10:])[0] / 2**16
    return x, y, z

def get_cXYZ(profile: bytes, colour: bytes):
    i = profile.index(colour+b'XYZ')
    offset = struct.unpack('>I', profile[i+4:i+8])[0]
    length = struct.unpack('>I', profile[i+8:i+12])[0]
    profile = profile[offset:offset+length]
    xyz = parse_s15Fixed16Number(profile[8:])
    return np.array(xyz)

def get_cTRC(profile: bytes, colour: bytes):
    i = profile.index(colour+b'TRC')
    offset = struct.unpack('>I', profile[i+4:i+8])[0]
    length = struct.unpack('>I', profile[i+8:i+12])[0]
    dlen = struct.unpack('>I', profile[offset+8:offset+12])[0]
    
    if dlen == 1: # u8Fixed8Number, uint8 int + uint8 frac
        ufixed8p8 = struct.unpack('>BB', profile[offset+12:offset+length])
        gamma = ufixed8p8[0] + ufixed8p8[1] / 2**8
        return gamma, 'gamma'
    else:
        profile = profile[offset+12:offset+length]
        curve = np.array(struct.unpack('>'+'H'*(len(profile)//2), profile))
        return curve, 'curve'

def table2gamma(table):
    x = np.linspace(0, 1, len(table))
    # 대충 닮게만 만들어도 잘 돼 한잔 해
    popt, _ = curve_fit(lambda x, gamma: x ** gamma, x, table/65535)
    return popt[0], 'gamma'

# 변환 행렬 생성
# 각각 R과 G, B를 생성하기 위해 각 색상별 X, Y, Z의 배합량 계산
rXYZ = get_cXYZ(profile, b'r')
gXYZ = get_cXYZ(profile, b'g')
bXYZ = get_cXYZ(profile, b'b')

# RGB -> XYZ 변환 행렬
RGB2XYZ = np.array([rXYZ, gXYZ, bXYZ]).T
# 역연산용 변환 행렬
XYZ2RGB = np.linalg.inv(RGB2XYZ)

# 감마 값/커브 테이블 추출
rTRC = get_cTRC(profile, b'r')
gTRC = get_cTRC(profile, b'g')
bTRC = get_cTRC(profile, b'b')
if rTRC[1] != 'gamma': rTRC = table2gamma(rTRC[0])
if gTRC[1] != 'gamma': gTRC = table2gamma(gTRC[0])
if bTRC[1] != 'gamma': bTRC = table2gamma(bTRC[0])

# 여기서부터 실제 값이 필요함, 예시: 주황색 RGBA 픽셀
pixel = np.array([1.0, 0.5, 0.2, 1.0])

# 감마 적용
pixel[0] = pixel[0] ** rTRC[0]
pixel[1] = pixel[1] ** gTRC[0]
pixel[2] = pixel[2] ** bTRC[0]

# 변환 행렬 적용
XYZ = RGB2XYZ.dot(pixel[:3])

# 알파값은 어따 쳐팔아먹은거야
XYZA = np.append(XYZ, pixel[3])

# 저장
print(XYZA)

# 역변환
RGB = XYZ2RGB.dot(XYZA[:3])

RGB[0] = RGB[0] ** (1/rTRC[0])
RGB[1] = RGB[1] ** (1/gTRC[0])
RGB[2] = RGB[2] ** (1/bTRC[0])

print(np.append(RGB, XYZA[3]))