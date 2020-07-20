'''

1选取大小为256*256，“LENA.bmp”进行离散傅里叶变换，并进行中心点平移。
在进行中心平移后，以直流分流所在中心为原点，分别获取25*25、50*50、100*100的正方形频率子带进行逆傅里叶变换，
分析不同大小的正方形频率子带对空域图像恢复有什么影响？
从而得知不同的频率子带包含有什么信息？
这种规律是否可以用于图像压缩？
请写出算法思路，主要程序段，并对实验结果进行分析。（10分）
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


lena_fft = np.fft.fft2(lena_org)

lena_fft_shift = np.fft.fftshift(lena_fft)

fft = np.log(np.abs(lena_fft))
fft_shift = np.log(np.abs(lena_fft_shift))


#25*25频率子带
#print(lena_fft)
iimg25 = np.fft.ifft2(lena_fft_shift,(25,25))

iimg25 = np.abs(iimg25)

#50*50频率子带
iimg50 = np.fft.ifft2(lena_fft_shift,(50,50))
iimg50 = np.abs(iimg50)

#100*100频率子带
iimg100 = np.fft.ifft2(lena_fft_shift,(100,100))
iimg100 = np.abs(iimg100)

#150*150
iimg150 = np.fft.ifft2(lena_fft_shift,(150,150))
iimg150 = np.abs(iimg150)

#200*200
iimg200 = np.fft.ifft2(lena_fft_shift,(200,200))
iimg200 = np.abs(iimg200)

fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(10, 5))

ax[0,0].imshow(lena_org, cmap=plt.cm.gray)
ax[0,0].axis('off')
ax[0,0].set_title("original")

ax[0,1].imshow(fft, cmap=plt.cm.gray)
ax[0,1].set_title('fft')
ax[0,1].axis("off")

ax[1,0].imshow(fft_shift, cmap=plt.cm.gray)
ax[1,0].axis('off')
ax[1,0].set_title("fft shift")

ax[1,1].imshow(iimg25, cmap=plt.cm.gray)
ax[1,1].axis('off')
ax[1,1].set_title("ifft25")

ax[2,0].imshow(iimg50, cmap=plt.cm.gray)
ax[2,0].axis('off')
ax[2,0].set_title("ifft50")

ax[2,1].imshow(iimg100, cmap=plt.cm.gray)
ax[2,1].axis('off')
ax[2,1].set_title("ifft100")

ax[3,0].imshow(iimg150, cmap=plt.cm.gray)
ax[3,0].axis('off')
ax[3,0].set_title("ifft150")

ax[3,1].imshow(iimg200, cmap=plt.cm.gray)
ax[3,1].axis('off')
ax[3,1].set_title("ifft200")

plt.tight_layout()
plt.show()