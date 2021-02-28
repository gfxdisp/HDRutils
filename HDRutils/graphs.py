import matplotlib.pyplot as plt
import numpy as np

def scatter_pixels(in_img, out_img, density=0.025):
	"""
	Randomly select pixels and plot input pixel value vs ouput pixel value

	:in_img: the input image, typically one of the RAW images
	:out_img: the output image to check linearity such as the merged HDR image
	"""

	assert in_img.shape == out_img.shape, 'The images need to have the same shape'
	pixel_mask = np.random.randn(*in_img.shape[:2]) < density

	in_img = in_img / in_img.max()
	out_img = out_img / out_img.max()
	colors = ((1,0,0,0.1), (0,1,0,1), (0,0,1,0.1))
	for i, c in enumerate(colors):
		plt.scatter(in_img[...,i][pixel_mask], out_img[...,i][pixel_mask],
					color=c, marker='.', s=1)
	plt.grid()
	plt.yscale('log')
	plt.xscale('log')

	min_pix = 1e-3
	plt.xlim(min_pix, 1)
	plt.ylim(min_pix, 1)
	plt.show()
