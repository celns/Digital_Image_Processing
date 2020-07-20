
'''
2、	在给定的应用中，一个空域均值滤波器模板被用于输入图像以减少噪声，
然后再用一个拉普拉斯模板来增强图像中的细节。
如果交换这两个步骤的顺序，结果是否会相同？请写出算法思路，主要程序段，并对实验结果进行分析。

'''

import skimage
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import histogram, equalize_hist
from skimage.util import random_noise
import numpy as np
import cv2

from skimage.filters import rank,laplace
from skimage.morphology import disk

lena_org = io.imread("../pic/LENA.bmp",as_gray=True)


lena_gaussian = random_noise(lena_org, mode='gaussian',mean=0.05)

selem = disk(3)
lena_mean = rank.mean(lena_gaussian, selem=selem)
#print(lena_mean.shape)

lena_mean_laplace = laplace(lena_mean, ksize=3)


lena_laplace = laplace(lena_gaussian,ksize=3)

print(lena_laplace)

#lena_laplace_mean = rank.mean(lena_laplace, selem=selem)

fig, ax = plt.subplots(ncols=5, figsize=(10, 5))

ax[0].imshow(lena_org, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title("original")

ax[1].imshow(lena_gaussian, cmap=plt.cm.gray)
ax[1].set_title('gaussian')
ax[1].axis("off")

ax[2].imshow(lena_mean, cmap=plt.cm.gray)
ax[2].axis('off')
ax[2].set_title("mean filter")

ax[3].imshow(lena_mean_laplace, cmap=plt.cm.gray)
ax[3].axis('off')
ax[3].set_title("Mean+Laplace")


ax[4].imshow(lena_laplace, cmap=plt.cm.gray)
ax[4].axis('off')
ax[4].set_title("Laplace+Mean")


plt.tight_layout()
plt.show()