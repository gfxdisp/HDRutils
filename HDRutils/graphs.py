import matplotlib.pyplot as plt
import numpy as np, logging

logger = logging.getLogger(__name__)

def scatter_pixels(in_img, out_img, density=0.01):
	"""
	Randomly select pixels and plot input pixel value vs ouput pixel value

	:in_img: the linear reference image such as RAW image or merged linear image
	:out_img: the output image to check linearity
	"""

	assert in_img.shape == out_img.shape, 'The images need to have the same shape'
	pixel_mask = np.abs(np.random.randn(*in_img.shape[:2])) < density

	# in_img = in_img / in_img.max()
	# out_img = out_img / out_img.max()
	colors = ((1,0,0,0.1), (0,1,0,1), (0,0,1,0.1))

	logger.info(f'Using {np.count_nonzero(pixel_mask)} pixels for scatter plot')
	fig, ax = plt.subplots()
	for i, c in enumerate(colors):
		ax.scatter(in_img[...,i][pixel_mask], out_img[...,i][pixel_mask],
					color=c, marker='.', s=1)
	ax.grid()
	ax.set_xlabel('Ground truth pixel value')
	ax.set_ylabel('Test image pixel value')
	ax.set_yscale('log')
	ax.set_xscale('log')

	return fig, ax
