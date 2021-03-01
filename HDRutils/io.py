import logging
import numpy as np
import imageio as io, rawpy

logger = logging.getLogger(__name__)

def imread_libraw(raw, color_space='sRGB', wb='camera'):
	if color_space == 'sRGB':
		color_space = rawpy.ColorSpace.sRGB
	elif color_space == 'raw':
		color_space = rawpy.ColorSpace.raw
	elif color_space == 'Adobe':
		color_space = rawpy.ColorSpace.Adobe
	else:
		raise Exception('Unknown color-space. Use sRGB, raw or Adobe.')

	auto_wb = False
	if wb == 'auto':
		auto_wb = True
		wb = raw.daylight_whitebalance
	elif wb == 'camera':
		wb = raw.camera_whitebalance
	elif wb == 'none':
		wb = np.ones(4)
	else:
		raise Exception('Unknown white-balance. Use auto, camera or none.')

	# Different demosaicing algorithms can be used. See this link:
	# https://letmaik.github.io/rawpy/api/enums.html#demosaicalgorithm
	# user_flip set to 0 because otherwise we cannot align RAW and demosaiced pixel values
	img = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16,
						  use_auto_wb=auto_wb, user_wb=tuple(wb),
						  user_flip=0, output_color=color_space,
						  highlight_mode=rawpy.HighlightMode.Clip)
	return img


def imread(file, libraw=True, color_space='sRGB', wb='camera'):
	"""
	Wrapper for io.imread() or rawpy.imread() depending on extension provided.

	:file: input file can be a regular image, HDR image (.exr, .hdr) or a
		   RAW file. Supported RAW formats are (.dng, .arw, .cr2, .nef)
	:libraw: boolean flag indicating that RAW image should be processed by libraw
	:color_space: output color_space (libraw argument)
	:wb: white-balance setting (libraw argument)
	:return: image as np array
	"""
	raw_ext = ('.dng', '.arw', '.cr2', '.nef')
	if file.lower().endswith((raw_ext)):
		# Raw formats may be optionally processed with libraw
		raw = rawpy.imread(file)
		if libraw:
			return imread_libraw(raw, color_space, wb)
		else:
			return raw.raw_image_visible
	else:
		# For other formats, just use imageio
		return io.imread(file)


def imwrite(file, img):
	"""
	Wrapper for io.imwrite with some inital sanity checks

	:file: output filename with correct extension
	:img: HDR image to write; datatype should ideally be floating point image
	"""
	if img.dtype in (np.uint8, np.uint16):
		logger.warning('It is very unlikely that the image is HDR since '
			f'it is encoded using {img.dtype}')
		if file.endswith('.png'):
			io.imwrite(file, img, format='PNG-FI')
	elif img.dtype in (np.float32, np.float64, np.float128):
		img = img.astype(np.float32)
		if file.endswith(('.exr', '.hdr')):
			# Imegio needs an additional flag to prevent clipping to (2**16 - 1)
			io.imwrite(file, img, flags=0x0001)
	else:
		logger.warning('Non-standard extension/datatype. Reverting to imageio default')
		io.imwrite(file, img)
