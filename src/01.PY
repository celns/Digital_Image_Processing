'''

1、	选取大小为256*256，“LENA.bmp”分别进行自编直方图均衡处理和自编直方图规则化处理，
写出算法思想，主要程序段，实验结果分析
'''


import skimage
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import histogram
import numpy as np

#求得图像cdf
def cumulative_distribution(image, nbins=256):
    hist, bin_centers = histogram(image, nbins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    print(img_cdf.shape)
    return img_cdf, bin_centers

#直方图均衡化
def equalize_hist(image, nbins=256):
    cdf, bin_centers = cumulative_distribution(image, nbins)
    out = np.interp(image.flat, bin_centers, cdf)
    return out.reshape(image.shape)
#直方图规定化
def specification_hist(image,  target, nbins=256):
    cdf, bin_centers = cumulative_distribution(image, nbins)
    tcdf,tbin_centers = cumulative_distribution(target, nbins)
#    tout = np.interp(target.flat, tbin_centers, tcdf)
    out = np.interp(image.flat, bin_centers, tcdf)
    return out.reshape(image.shape)

lena_org = io.imread("../pic/LENA.bmp",as_gray=True)

hist, hist_centers = histogram(lena_org,nbins=128,normalize=True)
lena_ehist = equalize_hist(lena_org)
ehist, ehist_centers = histogram(lena_ehist,nbins=128,normalize=True)

target = io.imread("../pic/NOISE.bmp",as_gray=True)
lena_spec = specification_hist(lena_org, target)
shist, shist_centers = histogram(lena_spec,nbins=128,normalize=True)


fig, ax = plt.subplots(ncols=2,nrows=3, figsize=(10, 5))

ax[0,0].imshow(lena_org, cmap=plt.cm.gray)
ax[0,0].axis('off')

ax[0,1].plot(hist_centers, hist, lw=2)
ax[0,1].set_title('Histogram of grey values')

ax[1,0].imshow(lena_ehist,cmap=plt.cm.gray)
ax[1,0].set_title('equalization')

ax[1,1].plot(ehist_centers, ehist, lw=2)
ax[1,1].set_title('equalization')

ax[2,0].imshow(lena_spec, cmap=plt.cm.gray)
ax[2,0].set_title('specify')

ax[2,1].plot(shist_centers, shist, lw=2)
ax[2,1].set_title('specify')


#plt.tight_layout()
plt.show()