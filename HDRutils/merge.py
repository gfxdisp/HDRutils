import logging, os, tqdm
from fractions import Fraction

import numpy as np
import rawpy, exifread

import HDRutils

logger = logging.getLogger(__name__)

def get_metadata(files, color_space, wb):
	"""
	Get metadata from EXIF files and rawpy
	
	:files: filenames containing the inpt images
	:color_space: Output color-space. Pick 1 of [sRGB, raw, Adobe]
	:wb: White-balance to use. Pick 1 of [camera, auto, none]
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
	data['saturation_point'] = raw.white_level - 128	# We see artifacts without this offset
	data['color_space'] = color_space
	data['wb'] = wb

	logger.info(f"Stack contains {len(files)} images of size: {data['h']}x{data['w']}")
	logger.info(f"Exp: {data['exp']}")
	logger.info(f"Gain: {data['gain']}")
	logger.info(f"aperture: {data['aperture']}")
	logger.info(f"Black-level: {data['black_level']}")
	logger.info(f"Saturation point: {data['saturation_point']}")
	logger.info(f"Color-space: {color_space}")
	logger.info(f"White-balance: {wb}")

	return data


def get_unsaturated(raw, saturation_threshold, img=None):
	"""
	Estimate a boolean mask to identify unsaturated pixels. The boolean
	mask returned is either single channel or 3-channel depending on whether
	the RGB image is passed (using parameter "img")

	:raw: Bayer image before demosaicing
	:bits: Bit-depth of the RAW image
	:img: RGB image after processing by libraw
	:return: boolean unsaturated mask
	"""

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


def merge(files, align=False, demosaic_first=True, color_space='sRGB', wb='camera'):
	"""
	Merge multiple SDR images into a single HDR image after demosacing. This
	is a wrapper function that extracts metadata and calls the appropriate
	function.

	:files: filenames containing the inpt images
	:align: align by estimation homography
	:demosaic_first: order of operations
	:color_space: Output color-space. Pick 1 of [sRGB, raw, Adobe]
	:wb: White-balance values. Pick 1 of [camera, auto, none]
	:return: Merged FP32 HDR image
	"""
	data = get_metadata(files, color_space, wb)

	if demosaic_first:
		HDR = imread_demosaic_merge(files, data, align)
	else:
		HDR = imread_merge_demosaic(files, data, align)
	return HDR


def imread_demosaic_merge(files, metadata, align):
	"""
	First postprocess using libraw and then merge RGB images. This function
	merges in an online way and can handle a large number of inputs with
	little memory.

	:files: filenames containing the inpt images
	:metadata: internally generated metadata dict
	:align: perform homography based alignment before merging
	:return: Merged FP32 HDR image
	"""

	logger.info('Demosaicing before merging.')
	# Check for saturation in shortest exposure
	shortest_exposure = np.argmin(metadata['exp'] * metadata['gain'] * metadata['aperture'])
	logger.info(f'Shortest exposure is {shortest_exposure}')

	if align:
		ref_idx = np.argsort(metadata['exp'] * metadata['gain']
							 * metadata['aperture'])[len(files)//2]
		ref_img = HDRutils.imread(files[ref_idx], color_space=metadata['color_space'], wb=metadata['wb'])

	num_saturated = 0
	num, denom = np.zeros((2, metadata['h'], metadata['w'], 3))
	for i, f in enumerate(tqdm.tqdm(files)):
		raw = rawpy.imread(f)
		img = HDRutils.io.imread_libraw(raw, color_space=metadata['color_space'], wb=metadata['wb'])
		if align and i != ref_idx:
			img = HDRutils.align(ref_img, img)

		# Ignore saturated pixels in all but shortest exposure
		if i == shortest_exposure:
			unsaturated = np.ones_like(img, dtype=bool) 
			num_sat = np.count_nonzero(np.logical_not(get_unsaturated(raw, metadata['saturation_point'], img=img))) / 3
		else:
			unsaturated = get_unsaturated(raw, metadata['saturation_point'], img=img)
		X_times_t = img / metadata['gain'][i] / metadata['aperture'][i]
		denom[unsaturated] += metadata['exp'][i]
		num[unsaturated] += X_times_t[unsaturated]

	HDR = (num / denom).astype(np.float32)

	if num_sat > 0:
		logger.warning(f"{num_sat/(metadata['h']*metadata['w']):.3f}% of pixels (n={num_sat}) are \
			saturated in the shortest exposure. The values for those pixels will be inaccurate.")

	return HDR


def imread_merge_demosaic(files, metadata, align, pattern='RGGB'):
	"""
	Merge RAW images before demosaicing. This function merges in an online
	way and can handle a large number of inputs with little memory.

	:files: filenames containing the inpt images
	:metadata: internally generated metadata dict
	:align: perform homography based alignment before merging
	:pattern: bayer pattern used in RAW images
	:return: Merged FP32 HDR image
	"""

	if align:
		raise NotImplementedError
	# Some sanity checks and logs related to colour-space and white-balnce
	logger.info('Merging before demosaicing.')
	if metadata['color_space'] != 'raw':
		logger.warning('Switching to RAW color-space since it is the only one ' \
			'supported in the current mode.')
	raw = rawpy.imread(files[0])
	if metadata['wb'] == 'auto':
		logger.warning('Auto white-balance not supported. Using daylight whitebalance')
		wb = np.array(raw.daylight_whitebalance)
	elif metadata['wb'] == 'camera':
		logger.warning('Make sure that white-balance was not set to "auto" while ' \
			'capturing the stack. Using white-balance of first image.')
		wb = np.array(raw.camera_whitebalance)
	else:
		raise NotImplementedError

	if pattern == 'RGGB':
		assert wb[1] == wb[3] or wb[3] == 0
		wb = wb[:3] / wb.max()
	else:
		raise NotImplementedError

	# Check for saturation in shortest exposure
	shortest_exposure = np.argmin(metadata['exp'] * metadata['gain'] * metadata['aperture'])
	logger.info(f'Shortest exposure is {shortest_exposure}')

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
			num_sat = np.count_nonzero(np.logical_not(get_unsaturated(raw, metadata['saturation_point'])))
		else:
			unsaturated = get_unsaturated(raw, metadata['saturation_point'])
		
		# Subtract black level for linearity
		img -= black_frame

		X_times_t = img / metadata['gain'][i] / metadata['aperture'][i]
		denom[unsaturated] += metadata['exp'][i]
		num[unsaturated] += X_times_t[unsaturated]
		
	HDR_bayer = (num / denom)

	# Libraw does not support 32-bit values. Use colour-demosaicing instead:
	# https://colour-demosaicing.readthedocs.io/en/latest/manual.html
	logger.info('Running bilinear demosaicing')
	import colour_demosaicing as cd
	HDR = cd.demosaicing_CFA_Bayer_bilinear(HDR_bayer, pattern=pattern)

	# White-balance
	HDR = (HDR * wb[np.newaxis, np.newaxis, :]).astype(np.float32)
	if num_sat > 0:
		logger.warning(f"{num_sat/(metadata['h']*metadata['w']):.3f}% of pixels (n={num_sat}) are \
			saturated in the shortest exposure. The values for those pixels will be inaccurate.")

	return HDR
