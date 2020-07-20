'''
tryment
opencv
'''

import skimage.io as io
import matplotlib.pyplot as plt
import cv2
import numpy as np


image_org = io.imread('../pic/LENA.bmp')


fil = np.array([[ 0,-1, 0],
                [ -1, 4, -1],
                [  0, -1, 0]])

res = cv2.filter2D(image_org,-1,fil)



dst, ax = plt.subplots(ncols=5, figsize=(10, 10))


ax[0].imshow(image_org)
ax[0].axis('off')

ax[1].imshow(res)
ax[1].axis('off')

plt.tight_layout()
plt.show()