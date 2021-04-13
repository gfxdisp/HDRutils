import logging
import numpy as np
import imageio as io, rawpy

logger = logging.getLogger(__name__)

def imread_libraw(raw, color_space='srgb'):
	wb = None
	color_space = color_space.lower()
	if color_space == 'srgb':
		color_space = rawpy.ColorSpace.sRGB
	elif color_space == 'raw':
		color_space = rawpy.ColorSpace.raw
		wb = (1, 1, 1, 1)
	elif color_space == 'adobe':
		color_space = rawpy.ColorSpace.Adobe
	elif color_space == 'xyz':
		color_space = rawpy.ColorSpace.XYZ
	else:
		raise Exception('Unknown color-space. Use sRGB, raw or Adobe.')

	# Different demosaicing algorithms can be used. See this link:
	# https://letmaik.github.io/rawpy/api/enums.html#demosaicalgorithm
	# user_flip set to 0 because otherwise we cannot align RAW and demosaiced pixel values
	img = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, user_wb=wb,
						  user_flip=0, output_color=color_space)
	return img


def imread(file, libraw=True, color_space='srgb', wb='camera'):
	"""
	Wrapper for io.imread() or rawpy.imread() depending on extension provided.

	:file: input file can be a regular image, HDR image (.exr, .hdr) or a
		   RAW file. Supported RAW formats are (.dng, .arw, .cr2, .nef)
	:libraw: boolean flag indicating that RAW image should be processed by libraw
	:color_space: output color_space (libraw argument)
	:return: image as np array
	"""
	raw_ext = ('.dng', '.arw', '.cr2', '.nef')
	ldr_ext = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
	hdr_ext = ('.exr', '.hdr')

	if file.lower().endswith(raw_ext):
		# Raw formats may be optionally processed with libraw
		raw = rawpy.imread(file)
		if libraw:
			return imread_libraw(raw, color_space)
		else:
			return raw.raw_image_visible
	else:
		# Imagio imread should handle most formats
		if file.lower().endswith(ldr_ext):
			logger.info('LDR image file format provided')
			if file.lower().endswith('.png'):
				# 16-bit pngs require an additional flag
				return io.imread(file, format='PNG-FI')
		elif file.lower().endswith(hdr_ext):
			logger.info('HDR image file format provided')
		else:
			logger.warning('Unknown image file format. Reverting to imageio imread')
		return io.imread(file)


def imwrite(file, img):
	"""
	Wrapper for io.imwrite with some inital sanity checks

	:file: output filename with correct extension
	:img: HDR image to write; datatype should ideally be floating point image
	"""
	if img.dtype in (np.uint8, np.uint16):
		logger.info(f'LDR image provided, it is encoded using {img.dtype}')
		if file.endswith('.png'):
			# 16-bit pngs require an additional flag
			io.imwrite(file, img, format='PNG-FI')
	elif img.dtype in (np.float32, np.float64, np.float128):
		img = img.astype(np.float32)
		if file.endswith(('.exr', '.hdr')):
			# Imegio needs an additional flag to prevent clipping to (2**16 - 1)
			io.imwrite(file, img, flags=0x0001)
	else:
		logger.warning('Unknown extension/datatype. Reverting to imageio imwrite')
		io.imwrite(file, img)
