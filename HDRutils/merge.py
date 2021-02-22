import logging, os, tqdm
from fractions import Fraction

import numpy as np
import imageio as io, rawpy, exifread

# rawpy documentation - https://letmaik.github.io/rawpy/api/

def get_metadata(files, log, color_space, wb):
	"""
	Get metadata from EXIF files and rawpy
	
	:files: filenames containing the inpt images
	:log: instance of logging for warnings and verbose information
	:color_space: Output color-space. Pick 1 of [sRGB, raw, Adobe]
	:wb: White-balance to use. Pick 1 of [auto, camera, average, none]
	:return: A dictonary containing all the metadata
	"""

	# Read exposure time, gain and aperture from EXIF data
	data = dict()
	data['exp'], data['gain'], data['aperture'] = np.empty((3, len(files)))

	for i, file in enumerate(files):
		with open(file, 'rb') as f:
			tags = exifread.process_file(f)
		if 'EXIF ExposureTime' in tags:
			data['exp'][i] = np.float32(Fraction(tags['EXIF ExposureTime'].printable))
		elif 'Image ExposureTime' in tags:
			data['exp'][i] = float(Fraction(tags['Image ExposureTime'].printable))
		else:
			raise Exception('Unable to read exposure time. Check EXIF data.')

		if 'EXIF ISOSpeedRatings' in tags:
			data['gain'][i] = float(tags['EXIF ISOSpeedRatings'].printable)/100
		elif 'Image ISOSpeedRatings' in tags:
			data['gain'][i] = float(tags['Image ISOSpeedRatings'].printable)/100
		else:
			raise Exception('Unable to read ISO. Check EXIF data.')

		# Aperture formula from https://www.omnicalculator.com/physics/aperture-area
		focal_length = float(Fraction(tags['EXIF FocalLength'].printable))
		f_number = float(Fraction(tags['EXIF FNumber'].printable))
		data['aperture'] = np.pi * (focal_length / 2 / f_number)**2

	# Get remaining data from rawpy
	raw = rawpy.imread(files[0])
	data['h'], data['w'] = raw.postprocess().shape[:2]
	data['black_level'] = raw.black_level_per_channel
	data['saturation_point'] = raw.white_point
	data['bits'] = np.log2(2**(int(data['saturation_point'] - 1)).bit_length())

	assert color_space in ['sRGB', 'raw', 'Adobe'],
		'Unreconized color space. For "color_space" pick 1 of: [sRGB, raw, Adobe]'
	if color_space == 'sRGB':
		data['colorspace'] = rawpy.ColorSpace.sRGB
	elif color_space == 'raw':
		data['colorspace'] = rawpy.ColorSpace.raw
	elif color_space == 'Adobe':
		data['colorspace'] = rawpy.ColorSpace.Adobe

	assert color_space in ['auto', 'camera', 'average', 'none'],
		'Unreconized color space. For "color_space" pick 1 of: [auto, camera, average, none]'
	if wb == 'auto':
		data['auto_wb'] = True
		data['whitebalance'] = raw.daylight_whitebalance
	else:
		data['auto_wb'] = False
	if wb == 'camera':
		data['whitebalance'] = raw.camera_whitebalance
		for f in files[1:]:
			assert rawpy.imread(f).camera_whitebalance == data['whitebalance'],
				'Images have different white-balance values. For "wb" pick 1 of: [auto, average, none]'
		data['whitebalance'] = np.array(data['whitebalance'])
	elif wb == 'average':
		# Use average wb across all images
		data['whitebalance'] = np.mean([rawpy.imread(f).camera_whitebalance for f in files])
	elif wb == 'none':
		data['whitebalance'] = np.ones(4) * 1024

	log.info(f'Stack contains {len(files)} images of size: {h}x{w}')
	log.info('Exp: ' + str(data['exp']))
	log.info('Gain: ' + str(data['gain']))
	log.info('aperture: ' + str(data['aperture']))
	log.info('White-level: ' + str(data['white_level']))
	log.info('Saturation point: ' + str(data['saturation_point']))
	log.info('Bit-depth: ' + str(data['bits']))

	return data

def get_unsaturated(raw, img, bits):
	"""
	Estimate a boolean mask to identify unsaturated pixels.

	:raw: Bayer image before demosaicing
	:img: RGB image after processing by libraw
	:bits: Bit-depth of the RAW image
	:return: boolean unsaturated mask
	"""

	# Use the RAW image to determine the saturation point
	# Determine the bit-depth
	

	saturation_threshold = 2**bits - 128
	unsaturated = np.logical_and.reduce(raw.raw_image_visible[0::2,0::2] < saturation_threshold,
										raw.raw_image_visible[1::2,0::2] < saturation_threshold,
										raw.raw_image_visible[0::2,1::2] < saturation_threshold,
										raw.raw_image_visible[1::2,1::2] < saturation_threshold)

	# A silly way to do 2x box-filter upsampling 
	unsaturated4 = np.zeros([unsaturated.shape[0]*2, unsaturated.shape[1]*2], dtype=bool)
	unsaturated4[0::2,0::2] = unsaturated
	unsaturated4[1::2,0::2] = unsaturated
	unsaturated4[0::2,1::2] = unsaturated
	unsaturated4[1::2,1::2] = unsaturated

	# The channel could also become saturated after white-balance
	saturation_threshold = (2**16 - 128)
	unsaturated4 = np.logical_and(unsaturated4, np.all( img <= saturation_threshold, axis=2 ) ) 

	unsaturated = np.repeat(unsaturated4[:,:,np.newaxis], 3, axis=-1)

	return unsaturated

def imread_merge(files, color_space='sRGB', wb='auto'):
	"""
	Merge multiple SDR images into a single HDR image after demosacing. This
	function works in an online way and can handle a large number of inputs.

	:files: filenames containing the inpt images
	:color_space: Output color-space. Pick 1 of [sRGB, raw, Adobe]
	:wb: White-balance values to use. Pick 1 of [auto, camera, average, none]
	:return: Merged FP32 HDR image
	"""
	log = logging.getLogger('merge')
	metadata = get_metadata(files, log, color_space, wb)

	# Check for saturation in shortest exposure
	shortest_exposure = np.argmin(metadata['exp'] * metadata['gain'] * metadata['aperture'])

	num_saturated = 0
	num, denom = np.zeros((2, metadata['h'], metadata['w'], 3))
	for i, f in enumerate(tqdm.tqdm(files)):
		raw = rawpy.imread(f)

		# Different demosaicing algorithms can be used. See this link:
		# https://letmaik.github.io/rawpy/api/enums.html#demosaicalgorithm
		# user_flip set to 0 because otherwise we cannot align RAW and demosaiced pixel values
		img = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16,
			use_auto_wb=metadata['auto_wb'], user_wb=metadata['whitebalance'],
			user_flip=0, output_color=metadata['colorspace'],
			highlight_mode=rawpy.HighlightMode.Clip )

		# Ignore saturated pixels in all but shortest exposure
		if i == shortest_exposure:
			unsaturated = np.ones_like(img, dtype=bool) 
			num_sat = np.count_nonzero(np.logical_not(get_unsaturated(raw, img, metadata['bits']))) / 3
		else:
			unsaturated = get_unsaturated(raw, img, metadata['bits'])
		X_times_t = img / metadata['gain'][i] / metadata['aperture'][i]
		denom[unsaturated] += metadata['exp'][i]
		num[unsaturated] += X_times_t[unsaturated]

	HDR = (num / denom).astype(np.float32)

	if num_sat > 0:
		log.warning(f'{num_sat/(metadata['h']*metadata['w']):.3f}% of pixels (n={num_sat}) are \
			saturated in the shortest exposure. The values for those pixels will be inaccurate.')

	return HDR

# TODO
def imread_merge_demosaic(files, color_space='sRGB', wb='auto'):
	"""
	Merge RAW images before demosaicing.

	:files: filenames containing the inpt images
	:color_space: Output color-space. Pick 1 of [sRGB, raw, Adobe]
	:wb: White-balance values to use. Pick 1 of [auto, camera, average, none]
	:return: Merged FP32 HDR image
	"""
	pass

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
	if img.dtype == np.uint16:
		assert file.endswith('.png'), 'Use .png for uint16 image'
		img = img.astype(np.float32)
	else:
		if img.dtype in (np.float64, np.float128):
			img = img.astype(np.float32)
		assert file.endswith('.exr'), 'Use .exr for floating point image'
		assert img.dtype == np.float32, f'Unrecognized data type: {img.dtype}'

	io.imwrite(file, img)