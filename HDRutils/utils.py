import logging, os, tqdm
from fractions import Fraction
import exifread

import numpy as np, cv2
import rawpy

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr

import HDRutils.io as io

logger = logging.getLogger(__name__)


def get_metadata(files, color_space='sRGB', sat_percent=0.98, black_level=0, exp=None, gain=None,
				 aperture=None):
	"""
	Get metadata from EXIF files and rawpy. If the image file contains no metadata, exposure
	time, gain and aperture need to supplied explicitly.
	
	:return: A dictonary containing all the metadata
	"""
	# Read exposure time, gain and aperture from EXIF data
	data = dict()
	N = len(files)

	try:
		data['exp'], data['gain'], data['aperture'] = np.empty((3, N))
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
	except Exception:
		# Manually populate metadata for non-RAW formats
		assert exp is not None, 'Unable to read metada from file, please supply manually'
		data['exp'] = np.array(exp)
		data['gain'] = np.ones(N) if gain is None else np.array(gain)
		data['aperture'] = np.ones(N) if aperture is None else np.array(aperture)
		assert len(exp) == N and len(data['gain']) == N and len(data['aperture']) == N, \
			'Mismatch in dimensions of metadata supplied'

	try:
		# Get remaining data from rawpy
		raw = rawpy.imread(files[0])
		data['raw_format'] = True
		data['h'], data['w'] = raw.postprocess(user_flip=0).shape[:2]
		data['black_level'] = np.array(raw.black_level_per_channel)
		# For some cameras, the provided white_level is incorrect
		data['saturation_point'] = raw.white_level*sat_percent
	except rawpy._rawpy.LibRawFileUnsupportedError:
		data['raw_format'] = False
		longest_exposure = np.argmax(data['exp'] * data['gain'] * data['aperture'])
		img = io.imread(files[longest_exposure])
		data['dtype'] = img.dtype
		assert len(img.shape) == 2, 'Provided files should not be demosaiced'
		data['h'], data['w'] = img.shape
		if img.dtype == np.float32:
			data['saturation_point'] = img.max()
		elif img.dtype == np.uint16:
			data['saturation_point'] = 2**16 - 1
		elif img.dtype == np.uint8:
			data['saturation_point'] = 2**8 - 1
		shortest_exposure = np.argmin(data['exp'] * data['gain'] * data['aperture'])
		img = io.imread(files[shortest_exposure])
		data['black_level'] = np.array([black_level]*4)
		if np.abs(img.min() - black_level) > data['saturation_point'] * 0.01:
			logger.warning(f'Using black level {black_level}. Double check this with camera docs.')

	data['color_space'] = color_space.lower()

	logger.info(f"Stack contains {N} images of size: {data['h']}x{data['w']}")
	logger.info(f"Exp: {data['exp']}")
	logger.info(f"Gain: {data['gain']}")
	logger.info(f"aperture: {data['aperture']}")
	logger.info(f"Black-level: {data['black_level']}")
	logger.info(f"Saturation point: {data['saturation_point']}")
	logger.info(f"Color-space: {color_space}")

	return data


def get_unsaturated(raw=None, saturation_threshold=None, img=None, saturation_threshold_img=None):
	"""
	Estimate a boolean mask to identify unsaturated pixels. The mask returned is either single
	channel or 3-channel depending on whether the RGB image is passed (using parameter "img")

	:raw: Bayer image before demosaicing
	:bits: Bit-depth of the RAW image
	:img: RGB image after processing by libraw
	:sat_percent: Saturation offset from reported white-point
	:return: Boolean unsaturated mask
	"""
	if raw is not None:
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
			return unsaturated4

	assert img is not None, 'Neither RAW nor RGB image is provided'
	# The channel could become saturated after white-balance
	unsaturated_all = np.all(img < saturation_threshold_img, axis=2)

	if raw is None:
		unsaturated = np.repeat(unsaturated_all[:,:,np.newaxis], 3, axis=-1)
	else:
		unsaturated4 = np.logical_and(unsaturated4, unsaturated_all)
		unsaturated = np.repeat(unsaturated4[:,:,np.newaxis], 3, axis=-1)

	return unsaturated


def estimate_exposures(files, metadata, sat_percent=0.98, noise_floor=10, percentile=90, invert_gamma=False):
	"""
	Exposure times may be inaccurate. Estimate the correct values by fitting a linear system.
	
	:files: Image filenames
	:gains: Internal camera metadata dictionary
	:sat_percent: Saturation offset from reported white-point
	:noise_floor: All pixels smaller than this will be ignored
	:percentile: Use a small percentage of all the pixels for the best estimation
	:return: Corrected exposure times
	"""
	Y = np.array([io.imread(f, libraw=False) for f in files], dtype=np.float32)
	if metadata['raw_format']:
		black_frame = np.tile(metadata['black_level'].reshape(2, 2),
							  (metadata['h']//2, metadata['w']//2))
		unsaturated = [get_unsaturated(img, metadata['saturation_point']) for img in Y]
	else:
		black_frame = 0
		sat_point = sat_percent * (2**(8*metadata['dtype']).itemsize - 1)
		unsaturated = [get_unsaturated(img=img, saturation_threshold_img=sat_point) for img in Y]

	# Mask out saturated and noisy pixels
	above_noise_floor = Y >= black_frame + noise_floor
	mask = np.logical_and(unsaturated, above_noise_floor).all(axis=0)
	threshold = np.percentile(Y[-1][mask], percentile)
	mask = np.logical_and(mask, Y[-1] > threshold)
	Y = np.maximum(Y - black_frame, 0)
	if invert_gamma:
		max_value = np.iinfo(metadata['dtype']).max
		Y = (Y / max_value)**(invert_gamma) * max_value
	L = np.log(np.stack([y[mask] / (t * a) \
						 for y, t, a in zip(Y, metadata['gain'], metadata['aperture'])]))
	num_exp, num_pix = L.shape
	assert num_pix > 0, 'No valid pixels found. Use closer exposure times'

	# Construct sparse linear system O.E = M
	logger.info(f'Constructing sparse matrix (O) and vector (M) using {num_pix} pixels')
	rows = np.float32(np.arange(0, (num_exp - 1)*num_pix, 0.5))
	cols, data = np.zeros((2, 2*(num_exp - 1)*num_pix), dtype=np.float32)
	for i in range(0, 2*(num_exp - 1)*num_pix, num_pix*2):
		c = i // (num_pix*2)
		cols[i:(i+num_pix*2):2] = c
		cols[(i+1):(i+num_pix*2):2] = c + 1
		# # Weights don't seem to make a difference
		# W = np.sqrt(1/(np.log(1+np.sqrt(1/np.exp(L[c])))**2 + \
		# 			   np.log(1+np.sqrt(1/np.exp(L[c + 1])))**2))
		# data[i:(i+num_pix*2):2] = W
		# data[(i+1):(i+num_pix*2):2] = -W
	data[::2] = 1
	data[1::2] = -1

	O = csc_matrix((data, (rows, cols)), shape=((num_exp - 1)*num_pix, num_exp))
	M = np.empty((num_exp - 1)*num_pix, dtype=np.float32)
	for i in range(0, (num_exp - 1)*num_pix, num_pix):
		c = i // num_pix
		# # Weights don't seem to make a difference
		# W = np.sqrt(1/(np.log(1+np.sqrt(1/np.exp(L[c])))**2 + \
		# 			   np.log(1+np.sqrt(1/np.exp(L[c + 1])))**2))
		# M[i:(i+num_pix)] = (L[c] - L[c + 1]) * W
		M[i:(i+num_pix)] = (L[c] - L[c + 1])

	# Solve the system and get linearise
	logger.info('Solving the sparse linear system')
	exp = lsqr(O, M)[0]
	exp = np.exp(exp - exp.min()) * metadata['exp'].min()
	logger.warning(f"Exposure times in EXIF: {metadata['exp']}, estimated exposures: {exp}")
	return exp


def encode(im1, im2):
	lin_max = np.max((im1, im2))
	lin_min = np.max((np.min((im1, im2)), 1e-10))

	# Do not stretch or compress histogram too much
	if lin_max/lin_min > 10000: lin_min = lin_max/10000
	if lin_max/lin_min < 1000: lin_min = lin_max/1000

	enc1 = np.log(im1/lin_min + 1) / np.log(lin_max/lin_min + 1) * 255
	enc2 = np.log(im2/lin_min + 1) / np.log(lin_max/lin_min + 1) * 255

	return enc1.astype(np.uint8), enc2.astype(np.uint8)

def find_homography(kp1, kp2, matches):
	matched_kp1 = np.zeros((len(matches), 1, 2), dtype=np.float32)
	matched_kp2 = np.zeros((len(matches), 1, 2), dtype=np.float32)

	for i in range(len(matches)):
		matched_kp1[i] = kp1[matches[i].queryIdx].pt
		matched_kp2[i] = kp2[matches[i].trainIdx].pt

	homography, _ = cv2.findHomography(matched_kp1, matched_kp2, cv2.RANSAC, 1)

	return homography

def align(ref, target, warped, downsample=None):
	"""
	Align a pair of images. Use feature matching and homography estimation to
	align. This works well for camera motion when scene depth is small.

	:ref: input reference image
	:target: target image to estimate homography
	:warped: image to be warped
	:downsample: when working with large images, memory considerations might
				 make it necessary to compute homography on downsampled images
	:return: warped target image
	"""
	logger = logging.getLogger('align')
	logger.info('Aligning images using homography')
	h, w = ref.shape[:2]
	if downsample:
		assert downsample > 1
		ref = cv2.resize(ref, (0, 0), fx=1/downsample, fy=1/downsample)
		target_r = cv2.resize(target, (0, 0), fx=1/downsample, fy=1/downsample)
	else:
		target_r = target

	logger.info('Using SIFT feature detector')
	try:
		detector = cv2.xfeatures2d.SIFT_create()
	except:
		detector = cv2.SIFT_create()
	bf = cv2.BFMatcher(crossCheck=True)

	enc_ref, enc_target = encode(ref, target_r)
	kp_ref, desc_ref = detector.detectAndCompute(enc_ref, None)
	kp, desc = detector.detectAndCompute(enc_target, None)

	if len(kp) > 100000:
		# https://github.com/opencv/opencv/issues/5700
		logger.info('Too many keypoints detected. Restricting to 100k keypoints per image.')
		kp, desc = kp[:100000], desc[:100000]
		kp_ref, desc_ref = kp_ref[:100000], desc_ref[:100000]
	matches = bf.match(desc, desc_ref)

	if len(matches) < 10:
		logger.error('Not enough matches, homography alignment failed')
		return warped
	else:
		logger.info(f'{len(matches)} matches found, using top 100')
	matches = sorted(matches, key=lambda x:x.distance)[:100]

	# img = cv2.drawMatches(enc_target, kp, enc_ref, kp_ref, matches, None)

	H = find_homography(kp, kp_ref, matches)
	if H.max() > 1000:
		logger.warning('Large value detected in homography. Estimation may have failed.')
	logger.info(f'Estimated homography: {H}')
	if len(warped.shape) == 2:
		# Bayer image
		logger.info('Warping bayer image')
		h, w = h//2, w//2
		warped[::2,::2] = cv2.warpPerspective(warped[::2,::2], H, (w, h))
		warped[::2,1::2] = cv2.warpPerspective(warped[::2,1::2], H, (w, h))
		warped[1::2,::2] = cv2.warpPerspective(warped[1::2,::2], H, (w, h))
		warped[1::2,1::2] = cv2.warpPerspective(warped[1::2,1::2], H, (w, h))
	else:
		logger.info('Warping RGB image')
		warped = cv2.warpPerspective(warped, H, (w, h))

	return warped
