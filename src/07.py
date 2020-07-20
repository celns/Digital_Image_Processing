'''

将平滑参数𝛔设为可变参数，用于控制LOG边缘检测算子，
通过实验说明平滑参数𝛔与边缘检测图像之间的关系


'''
'''
首先对图像做高斯滤波，然后再求其拉普拉斯（Laplacian）二阶导数。
即图像与 Laplacian of the Gaussian function 进行滤波运算。
最后，通过检测滤波结果的零交叉（Zero crossings）可以获得图像或物体的边缘。
此为Laplacian-of-Gaussian (LoG)算子。

'''

import skimage
import matplotlib.pyplot as plt
from skimage import io
from skimage.exposure import histogram, equalize_hist
from skimage.util import random_noise
import numpy as np
import cv2
import math
from scipy import signal


from skimage.filters import gaussian, laplace
from skimage.morphology import disk

#构造log算子


'''
#未能达到预期效果
def LoG(d,n):
    kernel = np.zeros((n,n))
    c = int((n-1)/2)
    for i in range(n):
        for j in range(n):
            kernel[i,j] = (((c-i)**2+(c-j)**2-2*d**2)/d**4*np.exp(-(((c-i)**2+(c-j)**2))/(2*d**2)))

    kernel = kernel +np.min(kernel)
    kernel = kernel / np.sum(kernel)
    return kernel
'''


lena_org = io.imread("../pic/LENA.bmp",as_gray=True)
print(lena_org.dtype)
'''
sigma1 = laplace(gaussian(lena_org,sigma=2),ksize=5)
sigma2 = laplace(gaussian(lena_org,sigma=1),ksize=5)
sigma3 = laplace(gaussian(lena_org,sigma=3),ksize=5)
sigma4 = laplace(gaussian(lena_org,sigma=4),ksize=5)
sigma5 = laplace(gaussian(lena_org,sigma=5),ksize=5)
'''
fil = np.array([[ 0,-1, 0],
                [ -1, 4, -1],
                [  0, -1, 0]])

sigma1 = cv2.filter2D(gaussian(lena_org,sigma=0.01),-1,fil)
sigma2 = cv2.filter2D(gaussian(lena_org,sigma=0.1),-1,fil)
sigma3 = cv2.filter2D(gaussian(lena_org,sigma=1),-1,fil)
sigma4 = cv2.filter2D(gaussian(lena_org,sigma=10),-1,fil)
sigma5 = cv2.filter2D(gaussian(lena_org,sigma=20),-1,fil)
print(sigma1.dtype)

#res = cv2.filter2D(lena_org,-1,LoG(10,11))


#print(LoG(10,11))


#显示
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10),sharex=True, sharey=True)
ax = axes.ravel()

titles = ['Original', 'sigma0.01','sigma0.1','sigma1.0','sigma10.0','sigma20.0']
imgs = [lena_org,  sigma1,  sigma2,  sigma3,  sigma4,  sigma5]
for n in range(0, len(imgs)):
    ax[n].imshow(imgs[n], cmap=plt.cm.gray)
    ax[n].set_title(titles[n])
    ax[n].axis('off')


plt.tight_layout()
plt.show()