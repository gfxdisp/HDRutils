import logging, os, tqdm
from fractions import Fraction

import numpy as np
import rawpy, exifread

import HDRutils.io as io
from HDRutils.deghosting import align

logger = logging.getLogger(__name__)

def get_metadata(files, color_space, sat_percent):
	"""
	Get metadata from EXIF files and rawpy
	
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
			raise Exception(f'Unable to read exposure time for {file}. Check EXIF data.')

		if 'EXIF ISOSpeedRatings' in tags:
			data['gain'][i] = float(tags['EXIF ISOSpeedRatings'].printable)/100
		elif 'Image ISOSpeedRatings' in tags:
			data['gain'][i] = float(tags['Image ISOSpeedRatings'].printable)/100
		else:
			raise Exception(f'Unable to read ISO. Check EXIF data for {file}.')

		# Aperture formula from https://www.omnicalculator.com/physics/aperture-area
		focal_length = float(Fraction(tags['EXIF FocalLength'].printable))
		f_number = float(Fraction(tags['EXIF FNumber'].printable))
		data['aperture'][i] = np.pi * (focal_length / 2 / f_number)**2

	# Get remaining data from rawpy
	raw = rawpy.imread(files[0])
	data['h'], data['w'] = raw.postprocess(user_flip=0).shape[:2]
	data['black_level'] = np.array(raw.black_level_per_channel)
	# For some cameras, the provided white_level is incorrect
	data['saturation_point'] = raw.white_level*sat_percent
	data['color_space'] = color_space.lower()

	logger.info(f"Stack contains {len(files)} images of size: {data['h']}x{data['w']}")
	logger.info(f"Exp: {data['exp']}")
	logger.info(f"Gain: {data['gain']}")
	logger.info(f"aperture: {data['aperture']}")
	logger.info(f"Black-level: {data['black_level']}")
	logger.info(f"Saturation point: {data['saturation_point']}")
	logger.info(f"Color-space: {color_space}")

	return data


def get_unsaturated(raw, saturation_threshold, img=None, sat_percent=None):
	"""
	Estimate a boolean mask to identify unsaturated pixels. The mask returned is either single
	channel or 3-channel depending on whether the RGB image is passed (using parameter "img")

	:raw: Bayer image before demosaicing
	:bits: Bit-depth of the RAW image
	:img: RGB image after processing by libraw
	:sat_percent: Saturation offset from reported white-point
	:return: Boolean unsaturated mask
	"""

	unsaturated = np.logical_and.reduce((raw[0::2,0::2] < saturation_threshold,
										raw[1::2,0::2] < saturation_threshold,
										raw[0::2,1::2] < saturation_threshold,
										raw[1::2,1::2] < saturation_threshold))

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
		saturation_threshold = (2**16 - 1) * sat_percent
		unsaturated4 = np.logical_and(unsaturated4, np.all(img < saturation_threshold, axis=2))
		unsaturated = np.repeat(unsaturated4[:,:,np.newaxis], 3, axis=-1)

	return unsaturated


def merge(files, do_align=False, demosaic_first=True, color_space='sRGB', wb=[1, 1, 1],
		  saturation_percent=0.98, normalize=False):
	"""
	Merge multiple SDR images into a single HDR image after demosacing. This is a wrapper
	function that extracts metadata and calls the appropriate function.

	:files: Filenames containing the inpt images
	:do_align: Align by estimation homography
	:demosaic_first: Order of operations
	:color_space: Output color-space. Pick 1 of [sRGB, raw, Adobe, XYZ]
	:wb: White-balance values after merging.
	:saturation_percent: Saturation offset from reported white-point
	:normalize: Output pixels lie between 0 and 1
	:return: Merged FP32 HDR image
	"""
	data = get_metadata(files, color_space, saturation_percent)

	if demosaic_first:
		HDR, num_sat = imread_demosaic_merge(files, data, do_align, saturation_percent)
	else:
		HDR, num_sat = imread_merge_demosaic(files, data, do_align)

	if num_sat > 0:
		logger.warning(f'{num_sat/(data["h"]*data["w"]):.3f}% of pixels (n={num_sat}) are ' \
					   'saturated in the shortest exposure. The values for these pixels will ' \
					   'be inaccurate.')

	if HDR.min() < 0:
		logger.info('Clipping negative pixels.')
		HDR[HDR < 0] = 0
	assert len(wb) == 3, 'Provide list [R G B] corresponding to white patch in the image'
	HDR = HDR * np.array(wb)[None, None, :]

	if normalize:
		HDR = HDR / HDR.max()
	return HDR.astype(np.float32)


def imread_demosaic_merge(files, metadata, do_align, sat_percent):
	"""
	First postprocess using libraw and then merge RGB images. This function merges in an online
	way and can handle a large number of inputs with little memory.
	"""

	logger.info('Demosaicing before merging.')
	
	# Check for saturation in shortest exposure
	shortest_exposure = np.argmin(metadata['exp'] * metadata['gain'] * metadata['aperture'])
	logger.info(f'Shortest exposure is {shortest_exposure}')

	if do_align:
		ref_idx = np.argsort(metadata['exp'] * metadata['gain']
							 * metadata['aperture'])[len(files)//2]
		ref_img = io.imread(files[ref_idx]) / metadata['exp'][ref_idx] \
											/ metadata['gain'][ref_idx] \
											/ metadata['aperture'][ref_idx]

	num_saturated = 0
	num, denom = np.zeros((2, metadata['h'], metadata['w'], 3))
	for i, f in enumerate(tqdm.tqdm(files)):
		raw = rawpy.imread(f)
		img = io.imread_libraw(raw, color_space=metadata['color_space'])
		if do_align and i != ref_idx:
			scaled_img = img / metadata['exp'][i] \
							 / metadata['gain'][i] \
							 / metadata['aperture'][i]
			img = align(ref_img, scaled_img, img)

		# Ignore saturated pixels in all but shortest exposure
		if i == shortest_exposure:
			unsaturated = np.ones_like(img, dtype=bool)
			num_sat = np.count_nonzero(np.logical_not(
				get_unsaturated(raw.raw_image_visible, metadata['saturation_point'],
								img, sat_percent))) / 3
		else:
			unsaturated = get_unsaturated(raw.raw_image_visible, metadata['saturation_point'],
										  img, sat_percent)
		X_times_t = img / metadata['gain'][i] / metadata['aperture'][i]
		denom[unsaturated] += metadata['exp'][i]
		num[unsaturated] += X_times_t[unsaturated]

	HDR = num / denom

	return HDR, num_sat


def imread_merge_demosaic(files, metadata, do_align, pattern='RGGB'):
	"""
	Merge RAW images before demosaicing. This function merges in an online
	way and can handle a large number of inputs with little memory.
	"""

	if do_align:
		ref_idx = np.argsort(metadata['exp'] * metadata['gain']
							 * metadata['aperture'])[len(files)//2]
		ref_img = io.imread(files[ref_idx]) / metadata['exp'][ref_idx] \
											/ metadata['gain'][ref_idx] \
											/ metadata['aperture'][ref_idx]

	logger.info('Merging before demosaicing.')
	raw = rawpy.imread(files[0])
	
	# More transforms available here:
	# http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
	if metadata['color_space'] == 'raw':
		color_mat = np.eye(3)
	else:
		assert (raw.rgb_xyz_matrix[-1] == 0).all()
		native2xyz = np.linalg.inv(raw.rgb_xyz_matrix[:-1])

		if metadata['color_space'] == 'xyz':
			xyz2out = np.eye(3)
		elif metadata['color_space'] == 'srgb':
			xyz2out = np.array([[3.2406, -1.5372, -0.4986],
								[-0.9689, 1.8758, 0.0415],
								[0.0557, -0.2040, 1.0570]])
		elif metadata['color_space'] == 'adobe':
			xyz2out = np.array([[2.0413690, -0.5649464, -0.3446944],
								[-0.9692660, 1.8760108, 0.0415560],
								[0.0134474, -0.1183897, 1.0154096]])
		else:
			logger.warning('Unsupported color-space, switching to camara raw.')
			native2xyz = np.eye(3)
			xyz2out = np.eye(3)
		color_mat = (xyz2out @ native2xyz).transpose()

	# Check for saturation in shortest exposure
	shortest_exposure = np.argmin(metadata['exp'] * metadata['gain'] * metadata['aperture'])
	logger.info(f'Shortest exposure is {shortest_exposure}')

	num_saturated = 0
	num, denom = np.zeros((2, metadata['h'], metadata['w']))
	black_frame = np.tile(metadata['black_level'].reshape(2, 2),
						  (metadata['h']//2, metadata['w']//2))
	for i, f in enumerate(tqdm.tqdm(files)):
		img = io.imread(f, libraw=False).astype(np.float32)
		if do_align and i != ref_idx:
			scaled_img = io.imread(f).astype(np.float32) / metadata['exp'][i] \
														 / metadata['gain'][i] \
														 / metadata['aperture'][i]
			img = align(ref_img, scaled_img, img)

		# Ignore saturated pixels in all but shortest exposure
		if i == shortest_exposure:
			unsaturated = np.ones_like(img, dtype=bool)
			num_sat = np.count_nonzero(np.logical_not(get_unsaturated(
				img, metadata['saturation_point'])))
		else:
			unsaturated = get_unsaturated(img, metadata['saturation_point'])
		
		# Subtract black level for linearity
		img -= black_frame

		X_times_t = img / metadata['gain'][i] / metadata['aperture'][i]
		denom[unsaturated] += metadata['exp'][i]
		num[unsaturated] += X_times_t[unsaturated]
		
	HDR_bayer = num / denom

	# Libraw does not support 32-bit values. Use colour-demosaicing instead:
	# https://colour-demosaicing.readthedocs.io/en/latest/manual.html
	logger.info('Running bilinear demosaicing')
	import colour_demosaicing as cd
	HDR = cd.demosaicing_CFA_Bayer_bilinear(HDR_bayer, pattern=pattern)

	# Convert to output color-space
	logger.info(f'Using color matrix: {color_mat}')
	HDR = HDR @ color_mat

	return HDR, num_sat
