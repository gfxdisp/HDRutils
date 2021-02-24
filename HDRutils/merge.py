import logging, os, tqdm
from fractions import Fraction

import numpy as np
import rawpy, exifread

# rawpy documentation - https://letmaik.github.io/rawpy/api/

def get_metadata(files, log, color_space, wb):
	"""
	Get metadata from EXIF files and rawpy
	
	:files: filenames containing the inpt images
	:log: instance of logging for warnings and verbose information
	:color_space: Output color-space. Pick 1 of [sRGB, raw, Adobe]
	:wb: White-balance to use. Pick 1 of [camera, auto, average, none]
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
		data['aperture'][i] = np.pi * (focal_length / 2 / f_number)**2

	# Get remaining data from rawpy
	raw = rawpy.imread(files[0])
	data['h'], data['w'] = raw.postprocess().shape[:2]
	data['black_level'] = np.array(raw.black_level_per_channel)
	data['saturation_point'] = raw.white_level
	data['bits'] = np.log2(2**(int(data['saturation_point'] - 1)).bit_length())

	assert color_space in ['sRGB', 'raw', 'Adobe'], \
		'Unreconized color space. For "color_space" pick 1 of: [sRGB, raw, Adobe]'
	if color_space == 'sRGB':
		data['colorspace'] = rawpy.ColorSpace.sRGB
	elif color_space == 'raw':
		data['colorspace'] = rawpy.ColorSpace.raw
	elif color_space == 'Adobe':
		data['colorspace'] = rawpy.ColorSpace.Adobe

	assert wb in ['camera', 'auto', 'average', 'none'], \
		'Unreconized white-balance. For "color_space" pick 1 of: [camera, auto, average, none]'
	if wb == 'auto':
		data['auto_wb'] = True
		data['whitebalance'] = raw.daylight_whitebalance
	else:
		data['auto_wb'] = False
	if wb == 'camera':
		data['whitebalance'] = raw.camera_whitebalance
		for f in files[1:]:
			assert rawpy.imread(f).camera_whitebalance == data['whitebalance'], \
				'Images have different white-balance values. Perhaps you want to pass "average"'
		data['whitebalance'] = np.array(data['whitebalance'])
	elif wb == 'average':
		# Use average wb across all images
		data['whitebalance'] = np.mean([rawpy.imread(f).camera_whitebalance for f in files], axis=0)
	elif wb == 'none':
		data['whitebalance'] = np.ones(4) * 1024

	log.info(f"Stack contains {len(files)} images of size: {data['h']}x{data['w']}")
	log.info(f"Exp: {data['exp']}")
	log.info(f"Gain: {data['gain']}")
	log.info(f"aperture: {data['aperture']}")
	log.info(f"Black-level: {data['black_level']}")
	log.info(f"Saturation point: {data['saturation_point']}")
	log.info(f"Estimated bit-depth: {data['bits']}")
	log.info(f"Color-space: {color_space}")
	if data['auto_wb']:
		log.info('White-balance: auto')
	else:
		log.info(f"White-balance: {data['whitebalance']}")

	return data


def get_unsaturated(raw, bits, img=None):
	"""
	Estimate a boolean mask to identify unsaturated pixels.

	:raw: Bayer image before demosaicing
	:bits: Bit-depth of the RAW image
	:img: RGB image after processing by libraw
	:return: boolean unsaturated mask
	"""

	saturation_threshold = 2**bits - 128
	unsaturated = np.logical_and.reduce((raw.raw_image_visible[0::2,0::2] < saturation_threshold,
										raw.raw_image_visible[1::2,0::2] < saturation_threshold,
										raw.raw_image_visible[0::2,1::2] < saturation_threshold,
										raw.raw_image_visible[1::2,1::2] < saturation_threshold))

	# A silly way to do 2x box-filter upsampling 
	unsaturated4 = np.zeros([unsaturated.shape[0]*2, unsaturated.shape[1]*2], dtype=bool)
	unsaturated4[0::2,0::2] = unsaturated
	unsaturated4[1::2,0::2] = unsaturated
	unsaturated4[0::2,1::2] = unsaturated
	unsaturated4[1::2,1::2] = unsaturated

	if img is None:
		unsaturated = unsaturated4
	else:
		# The channel could also become saturated after white-balance
		saturation_threshold = (2**16 - 128)
		unsaturated4 = np.logical_and(unsaturated4, np.all(img <= saturation_threshold, axis=2))
		unsaturated = np.repeat(unsaturated4[:,:,np.newaxis], 3, axis=-1)

	return unsaturated


def merge(files, demosaic_first=True, color_space='sRGB', wb='camera'):
	"""
	Merge multiple SDR images into a single HDR image after demosacing. This
	function merges in an online way and can handle a large number of inputs.

	:files: filenames containing the inpt images
	:demosaic_first: order of operations
	:color_space: Output color-space. Pick 1 of [sRGB, raw, Adobe]
	:wb: White-balance values to use. Pick 1 of [camera, auto, average, none]
	:return: Merged FP32 HDR image
	"""
	log = logging.getLogger('merge')
	data = get_metadata(files, log, color_space, wb)

	if demosaic_first:
		HDR = demosaic_merge(files, data, log)
	else:
		HDR = merge_demosaic(files, data, log)
	return HDR

def demosaic_merge(files, metadata, log):
	"""
	First postprocess using libraw and then merge RGB images
	"""
	log.info('Demosaicing before merging.')
	# Check for saturation in shortest exposure
	shortest_exposure = np.argmin(metadata['exp'] * metadata['gain'] * metadata['aperture'])
	log.info(f'Shortest exposure is {shortest_exposure}')

	num_saturated = 0
	num, denom = np.zeros((2, metadata['h'], metadata['w'], 3))
	for i, f in enumerate(tqdm.tqdm(files)):
		raw = rawpy.imread(f)

		# Different demosaicing algorithms can be used. See this link:
		# https://letmaik.github.io/rawpy/api/enums.html#demosaicalgorithm
		# user_flip set to 0 because otherwise we cannot align RAW and demosaiced pixel values
		img = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16,
			use_auto_wb=metadata['auto_wb'], user_wb=tuple(metadata['whitebalance']),
			user_flip=0, output_color=metadata['colorspace'],
			highlight_mode=rawpy.HighlightMode.Clip)

		# Ignore saturated pixels in all but shortest exposure
		if i == shortest_exposure:
			unsaturated = np.ones_like(img, dtype=bool) 
			num_sat = np.count_nonzero(np.logical_not(get_unsaturated(raw, metadata['bits'], img=img))) / 3
		else:
			unsaturated = get_unsaturated(raw, metadata['bits'], img=img)
		X_times_t = img / metadata['gain'][i] / metadata['aperture'][i]
		denom[unsaturated] += metadata['exp'][i]
		num[unsaturated] += X_times_t[unsaturated]

	HDR = (num / denom).astype(np.float32)

	if num_sat > 0:
		log.warning(f"{num_sat/(metadata['h']*metadata['w']):.3f}% of pixels (n={num_sat}) are \
			saturated in the shortest exposure. The values for those pixels will be inaccurate.")

	return HDR


def merge_demosaic(files, metadata, log, pattern='RGGB'):
	"""
	Merge RAW images before demosaicing.
	"""
	import colour_demosaicing as cd

	# Some sanity checks and logs related to colour-space and white-balnce
	log.info('Merging before demosaicing.')
	if metadata['colorspace'] != rawpy.ColorSpace.raw:
		log.warning('Switching to RAW color-space since it is the only one ' \
			'supported in the current mode.')
	wb = metadata['whitebalance']
	if pattern == 'RGGB':
		assert wb[1] == wb[3] or wb[3] == 0
		wb = wb[:3] / wb.max()
	else:
		log.warning('Untested pattern. Make sure your white-balance is correct.')

	# Check for saturation in shortest exposure
	shortest_exposure = np.argmin(metadata['exp'] * metadata['gain'] * metadata['aperture'])
	log.info(f'Shortest exposure is {shortest_exposure}')

	num_saturated = 0
	num, denom = np.zeros((2, metadata['h'], metadata['w']))
	black_frame = np.tile(metadata['black_level'].reshape(2, 2),
		(metadata['h']//2, metadata['w']//2))
	for i, f in enumerate(tqdm.tqdm(files)):
		raw = rawpy.imread(f)

		img = raw.raw_image_visible.astype(np.float32)

		# Ignore saturated pixels in all but shortest exposure
		if i == shortest_exposure:
			unsaturated = np.ones_like(raw, dtype=bool) 
			num_sat = np.count_nonzero(np.logical_not(get_unsaturated(raw, metadata['bits'])))
		else:
			unsaturated = get_unsaturated(raw, metadata['bits'])
		
		# Subtract black level for linearity
		img -= black_frame

		X_times_t = img / metadata['gain'][i] / metadata['aperture'][i]
		denom[unsaturated] += metadata['exp'][i]
		num[unsaturated] += X_times_t[unsaturated]
		
	HDR_bayer = (num / denom)

	# Libraw does not support 32-bit values. Use colour-demosaicing instead:
	# https://colour-demosaicing.readthedocs.io/en/latest/manual.html
	log.info('Running bilinear demosaicing')
	HDR = cd.demosaicing_CFA_Bayer_bilinear(HDR_bayer, pattern='RGGB')

	# White-balance
	HDR = (HDR * wb[np.newaxis, np.newaxis, :]).astype(np.float32)
	if num_sat > 0:
		log.warning(f"{num_sat/(metadata['h']*metadata['w']):.3f}% of pixels (n={num_sat}) are \
			saturated in the shortest exposure. The values for those pixels will be inaccurate.")

	return HDR
