import struct, zlib, json, os, sys
import numpy as np
from decimal import Decimal

colour_spaces = {
    'RGB': {
        'sRGB': b'\x00\x00',
        'Adobe': b'\x00\x01',
        'ProPhoto': b'\x00\x02',
        'P3': b'\x00\x03',
        'NTSC': b'\x00\x04',
        'PAL': b'\x00\x05',
    },
    'XYZ': {
        'CIE 1931': b'\x00\x00',
        'CIE 1964': b'\x00\x01',
    },
    'YUV': {
        'YCbCr': b'\x00\x00',
        'YUV': b'\x00\x01',
        'YPbPr': b'\x00\x02',
    },
    'HSV': {
        'HSV': b'\x00\x00',
        'HSL': b'\x00\x01',
    },
    'CMYK': {
        'CMYK': b'\x00\x00',
        'CMY': b'\x00\x01',
    }
}

def layer(ltype, ljson: dict, opacity: float, z_index, palette, cspace, resolution, scale: tuple = (0, 0), offset: tuple = (0, 0), rotation: Decimal = 0, lock: bool = False, hidden: bool = False):
    ljson = json.dumps(ljson, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    ltype = ltype == 'image' and b'\x00' or ltype=='text' and '\xff'
    opacity = struct.pack('>e', opacity)
    z_index = struct.pack('>H', z_index)
    vl = struct.pack('>B', (hidden and 1 or 0)<<7 | (lock and 1 or 0)<<6)

    palt = struct.pack('>B', ['XYZ', 'RGB', 'YUV', 'HSV', 'CMYK'].index(palette))

    if palette == 'XYZ':
        cspace = b'\x00\x00'
    elif cspace == 'json':
        cspace = b'\xff\xff'
    else: cspace = colour_spaces[palette][cspace]

    offset = list(offset)
    try: offset[0] = (int(Decimal(offset[0])), int(str(Decimal(offset[0])).split('.')[1][:2]))
    except: offset[0] = (int(Decimal(offset[0])), 0)
    try: offset[1] = (int(Decimal(offset[1])), int(str(Decimal(offset[1])).split('.')[1][:2]))
    except: offset[1] = (int(Decimal(offset[1])), 0)

    try: rotation = (int(Decimal(rotation)), int(str(Decimal(rotation)).split('.')[1][:9]))
    except: rotation = (int(Decimal(rotation)), 0)

    rint = struct.pack('>H', rotation[0]<<7)
    rdec = struct.pack('>I', rotation[1]<<1)
    rotation = rint[0:1]+struct.pack('>B', rint[1]|rdec[0])+rdec[1:]
    # rint = int.from_bytes(rotation, 'big')>>31
    # rdec = int.from_bytes(rotation, 'big')-(rint<<31)>>1

    return bytes(
        b'\xff\xf9\x72\x7a'+
        struct.pack('>I', len(ljson))+
        ltype+opacity+z_index+palt+cspace+
        vl+b'\x00'*7+
        struct.pack('>II', *resolution)+
        struct.pack('>ee', *scale)+b'\x00'*4+
        int.to_bytes(offset[0][0], 3, 'big')+struct.pack('>B', offset[0][1])+
        int.to_bytes(offset[1][0], 3, 'big')+struct.pack('>B', offset[1][1])+
        rotation+b'\x00'*7+
        struct.pack('>I', zlib.crc32(ljson))+

        ljson
    )

def horizimg_header(layerno, colno, data: bytes):
    return bytes(
        b'\xff\xfd\x69\xc6'+
        struct.pack('>H', layerno)+
        b'\x00'*2+
        struct.pack('>I', colno)+
        b'\x00'*4+
        struct.pack('>Q', len(data))+
        b'\x00'*4+
        struct.pack('>I', zlib.crc32(data))
    )