'''
4、	将10个离散点（1, 0），（2, 3），（4, 6），（3, 2），（5, 1），（6， 3），（7, 4），（8, 5），（9, 4），（10, 2）
投射到直角坐标系中，然后在hough空间内找到共线交点，
利用自编hough变换求出这10个离散点的边缘直线段。
并将下面一幅图像分别利用prewitt, otsu, Hough变换三种方法求出图像的边缘，
请写出算法思路，主要程序段，并对实验结果进行分析。
'''
import skimage
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import histogram, equalize_hist
from skimage.util import random_noise
import numpy as np
import cv2

from skimage.transform import hough_line
from skimage.filters import rank,laplace
from skimage.morphology import disk

points = np.zeros((11,11))
points[1,0] = 1
points[2,3] = 1
points[4,6] = 1
points[3,2] = 1
points[5,1] = 1
points[6,3] = 1
points[7,4] = 1
points[8,5] = 1
points[9,4] = 1
points[10,2] = 1


out, angles, d = hough_line(points)






#可视化
fix, axes = plt.subplots(2, 2, figsize=(7, 4))

axes[0,0].imshow(points, cmap=plt.cm.gray)
axes[0,0].set_title('Input image')

axes[0,1].imshow(
    out, cmap=plt.cm.bone,
    extent=(np.rad2deg(angles[-1]), np.rad2deg(angles[0]), d[-1], d[0]))
axes[0,1].set_title('Hough transform')
axes[0,1].set_xlabel('Angle (degree)')
axes[0,1].set_ylabel('Distance (pixel)')

plt.tight_layout()
plt.show()

