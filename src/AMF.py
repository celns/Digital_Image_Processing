'''

AMF模块
'''
import numpy as np

def Adaptor(img, coord_x, coord_y, minSize, maxSize):
    filter_size = minSize
    offset = filter_size // 2

    roi = img[coord_x-offset:coord_x+offset+1, coord_y-offset:coord_y+offset+1]
    Zmin = np.min(roi)
    Zmax = np.max(roi)
    Zmed = np.median(roi)
    Zxy = img[coord_x, coord_y]

    if (Zmed > Zmin) and (Zmed < Zmax):
        if (Zxy > Zmin) and (Zxy < Zmax):
            return Zxy
        else:
            return Zmed
    else:
        filter_size = filter_size + 2
        if (filter_size <= maxSize):
            return Adaptor(img, coord_x, coord_y, filter_size, maxSize)
        else:
            return Zmed


def AMF(img, minSize, maxSize):
    offset = maxSize//2
    for x in range(offset, img.shape[0]-offset):
        for y in range(offset, img.shape[1]-offset):
            img[x,y] = Adaptor(img, x, y, minSize, maxSize)
    # amf_img = img[offset:img.shape[0]-offset, img.shape[1]-offset]
    return img