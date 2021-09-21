import logging, tqdm

import numpy as np
import rawpy
import colour_demosaicing as cd

import HDRutils.io as io
from HDRutils.utils import *

logger = logging.getLogger(__name__)


def merge(files, do_align=False, demosaic_first=True, normalize=False, color_space='sRGB',
		  wb=[1, 1, 1], saturation_percent=0.98, black_level=0, bayer_pattern='RGGB',
		  exp=None, gain=None, aperture=None, estimate_exp=True):
	"""
	Merge multiple SDR images into a single HDR image after demosacing. This is a wrapper
	function that extracts metadata and calls the appropriate function.

	:files: Filenames containing the inpt images
	:do_align: Align by estimation homography
	:demosaic_first: Order of operations
	:color_space: Output color-space. Pick 1 of [sRGB, raw, Adobe, XYZ]
	:normalize: Output pixels lie between 0 and 1
	:wb: White-balance values after merging.
	:saturation_percent: Saturation offset from reported white-point
	:black_level: Camera's black level
	:bayer_patter: Color filter array pattern of the camera
	:exp: Exposure time (in seconds) required when metadata is not present
	:gain: Camera gain (ISO/100) required when metadata is not present
	:aperture: Aperture required when metadata is not present
	:estimate_exp: Estimate exposure times by solving a linear system
	:return: Merged FP32 HDR image
	"""
	data = get_metadata(files, color_space, saturation_percent, black_level, exp, gain, aperture)
	if estimate_exp:
		data['exp'] = estimate_exposures(files, data)

	if demosaic_first:
		HDR, num_sat = imread_demosaic_merge(files, data, do_align, saturation_percent)
	else:
		HDR, num_sat = imread_merge_demosaic(files, data, do_align, bayer_pattern)

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
	assert metadata['raw_format'], 'Libraw unsupported, use merge(..., demosaic_first=False)'
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
		saturation_point_img = sat_percent * (2**(8*img.dtype.itemsize) - 1)
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
								img, saturation_point_img))) / 3
		else:
			unsaturated = get_unsaturated(raw.raw_image_visible, metadata['saturation_point'],
										  img, saturation_point_img)
		X_times_t = img / metadata['gain'][i] / metadata['aperture'][i]
		denom[unsaturated] += metadata['exp'][i]
		num[unsaturated] += X_times_t[unsaturated]

	HDR = num / denom

	return HDR, num_sat


def imread_merge_demosaic(files, metadata, do_align, pattern):
	"""
	Merge RAW images before demosaicing. This function merges in an online
	way and can handle a large number of inputs with little memory.
	"""
	if do_align:
		ref_idx = np.argsort(metadata['exp'] * metadata['gain']
							 * metadata['aperture'])[len(files)//2]
		ref_img = io.imread(files[ref_idx]).astype(np.float32)
		if not metadata['raw_format']:
			ref_img = cd.demosaicing_CFA_Bayer_bilinear(ref_img, pattern=pattern)
		ref_img = ref_img / metadata['exp'][ref_idx] \
						  / metadata['gain'][ref_idx] \
						  / metadata['aperture'][ref_idx]

	logger.info('Merging before demosaicing.')

	# More transforms available here:
	# http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
	if metadata['color_space'] == 'raw':
		color_mat = np.eye(3)
	else:
		assert metadata['raw_format'], \
			'Only RAW color_space supported. Use merge(..., color_space=\'raw\')'
		raw = rawpy.imread(files[0])
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
			i_img = io.imread(f).astype(np.float32)
			if metadata['raw_format']:
				i_img = cd.demosaicing_CFA_Bayer_bilinear(i_img, pattern=pattern)
			i_img = i_img / metadata['exp'][i] \
						  / metadata['gain'][i] \
						  / metadata['aperture'][i]
			img = align(ref_img, i_img, img)

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
	HDR = cd.demosaicing_CFA_Bayer_bilinear(HDR_bayer, pattern=pattern)

	# Convert to output color-space
	logger.info(f'Using color matrix: {color_mat}')
	HDR = HDR @ color_mat

	return HDR, num_sat
