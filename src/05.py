'''


3、	对一幅灰度图像“LENA.bmp”分别添加椒盐噪声和方差为0.02的高斯噪声，
再使用自编均值滤波器，自编中值滤波器和自编自适应中值滤波器对图像进行处理，
请写出算法思路，主要程序段，并对实验结果进行分析
'''



import skimage
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import histogram, equalize_hist
from skimage.util import random_noise
import numpy as np
from AMF import AMF

from skimage.filters import rank,median
from skimage.morphology import disk

noise_org = io.imread("../pic/NOISE.bmp",as_gray=True)
noise_roi = noise_org[100:200,100:200]

selem = disk(3)
noise_mean = rank.mean(noise_roi, selem=selem)

hist, hist_centers = histogram(noise_org,nbins=256,normalize=True)
noise_median = median(noise_roi,selem = selem)




fig, ax = plt.subplots(ncols=3,nrows=2, figsize=(10, 5))

ax[0,0].imshow(noise_org, cmap=plt.cm.gray)
ax[0,0].axis('off')
ax[0,0].set_title("original")

ax[0,1].imshow(noise_roi, cmap=plt.cm.gray)
ax[0,1].axis('off')
ax[0,1].set_title("ROI")

ax[0,2].plot(hist_centers, hist, lw=2)
ax[0,2].set_title('Histogram')

ax[1,0].imshow(noise_mean, cmap=plt.cm.gray)
ax[1,0].axis('off')
ax[1,0].set_title("mean filter")

ax[1,1].imshow(noise_median, cmap=plt.cm.gray)
ax[1,1].axis('off')
ax[1,1].set_title("median")

plt.tight_layout()
plt.show()