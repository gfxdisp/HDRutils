import numpy as np
import imageio as io

def imread(file):
	"""
	Wrapper for io.imread(). Included for completeness

	:file: input HDR file (typically in OpenEXR format with extension .exr)
	"""
	return io.imread(file)

def imwrite(file, img):
	"""
	Wrapper for io.imwrite with some inital sanity checks

	:file: output filename with correct extension
	:img: HDR image to write(); datatype should be 16-bit integer or floating point (any precision)
	"""
	assert img.dtype != np.uint8, 'It is very unlikely that an HDR image has bit-depth of 8'
	if img.dtype == np.uint16:
		assert file.endswith('.png'), 'Use .png for uint16 image'
		io.imwrite(file, img, format='PNG-FI')
	else:
		if img.dtype in (np.float64, np.float128):
			img = img.astype(np.float32)
		assert file.endswith('.exr'), 'Use .exr for floating point image'
		assert img.dtype == np.float32, f'Unrecognized data type: {img.dtype}'

		# Imegio needs an additional flag to prevent clipping to (2**16 - 1)
		io.imwrite(file, img, flags=0x0001)