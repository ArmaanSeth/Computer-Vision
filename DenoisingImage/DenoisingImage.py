from skimage import io
from scipy import ndimage as nd
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_nl_means, estimate_sigma


img=io.imread('./noisy-image.png')
# guassianImg=nd.gaussian_filter(img, sigma=3)
# plt.imsave('guassian.jpeg', guassianImg)

medianImg=nd.median_filter(img, size=3)
# plt.imsave('median.jpeg', medianImg)
sigma_est=np.mean(estimate_sigma(img, multichannel=False))

denoise=denoise_nl_means(img, h=1.15*sigma_est, patch_size=5, fast_mode=False, patch_distance=3, multichannel=False)
plt.imsave('denoisy.jpeg', denoise)