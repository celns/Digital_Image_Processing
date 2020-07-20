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
from AMF import AMF #参见AMF.py

from skimage.filters import rank,median
from skimage.morphology import disk




lena_org = io.imread("../pic/LENA.bmp",as_gray=True)

#0.1的椒盐
lena_salt_pepper = random_noise(lena_org, mode = 's&p',salt_vs_pepper=0.1)
#0.02的高斯
lena_gaussian_sp = random_noise(lena_salt_pepper,mode='gaussian',mean=0.02)

selem = disk(2)

lena_mean = rank.mean(lena_gaussian_sp, selem=selem)

lena_median = median(lena_gaussian_sp,selem=selem)



fig, ax = plt.subplots(ncols=5, figsize=(10, 5))

ax[0].imshow(lena_org, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title("original")

ax[1].imshow(lena_gaussian_sp,cmap=plt.cm.gray)
ax[1].set_title('gaussian+salt+pepper')
ax[1].axis("off")

ax[2].imshow(lena_mean, cmap=plt.cm.gray)
ax[2].axis('off')
ax[2].set_title("mean filter")

ax[3].imshow(lena_median, cmap=plt.cm.gray)
ax[3].axis('off')
ax[3].set_title("median filter")

lena_AMF = AMF(lena_gaussian_sp, 3, 7)
ax[4].imshow(lena_AMF, cmap=plt.cm.gray)
ax[4].axis('off')
ax[4].set_title("AMF")

plt.tight_layout()
plt.show()