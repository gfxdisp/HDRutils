import logging
from fractions import Fraction
import exifread

import numpy as np, rawpy

import HDRutils.io as io

logger = logging.getLogger(__name__)


def get_metadata(files, exp, gain, aperture, color_space='sRGB', sat_percent=0.98, black_level=0, bits=None):
	"""
	Get metadata from EXIF files and rawpy. If the image file contains no metadata, exposure
	time, gain and aperture need to supplied explicitly.
	
	:return: A dictonary containing all the metadata
	"""
	# Read exposure time, gain and aperture from EXIF data
	data = dict()
	data['N'] = len(files)
	assert exp is None or len(exp) == data['N'], 'Incorrect exposure times provided'
	assert gain is None or len(gain) == data['N'], 'Incorrect gains times provided'
	assert aperture is None or isinstance(aperture, int) or isinstance(aperture, float) \
		   or len(aperture) == data['N']

	try:
		data['exp'], data['gain'], data['aperture'] = np.empty((3, data['N']))
		for i, file in enumerate(files):
			with open(file, 'rb') as f:
				tags = exifread.process_file(f)
			if exp is not None:
				data['exp'][i] = exp[i]
			elif 'EXIF ExposureTime' in tags:
				data['exp'][i] = np.float32(Fraction(tags['EXIF ExposureTime'].printable))
			elif 'Image ExposureTime' in tags:
				data['exp'][i] = float(Fraction(tags['Image ExposureTime'].printable))
			else:
				raise Exception(f'Unable to read exposure time for {file}. Check EXIF data.')

			if gain is not None:
				data['gain'][i] = gain[i]
			elif 'EXIF ISOSpeedRatings' in tags:
				data['gain'][i] = float(tags['EXIF ISOSpeedRatings'].printable)/100
			elif 'Image ISOSpeedRatings' in tags:
				data['gain'][i] = float(tags['Image ISOSpeedRatings'].printable)/100
			else:
				raise Exception(f'Unable to read ISO. Check EXIF data for {file}.')

			# Aperture formula from https://www.omnicalculator.com/physics/aperture-area
			if aperture:
				data['aperture'][i] = aperture
			else:
				focal_length = float(Fraction(tags['EXIF FocalLength'].printable))
				f_number = float(Fraction(tags['EXIF FNumber'].printable))
				data['aperture'][i] = np.pi * (focal_length / 2 / f_number)**2
	except Exception:
		# Manually populate metadata for non-RAW formats
		assert exp is not None, 'Unable to read metada from file, please supply manually'
		data['exp'] = np.array(exp)
		data['gain'] = np.ones(data['N']) if gain is None else np.array(gain)
		data['aperture'] = np.ones(data['N']) if aperture is None else np.array(aperture)
		assert len(exp) == data['N'] and len(data['gain']) == data['N'] and len(data['aperture']) == data['N'], \
			'Mismatch in dimensions of metadata supplied'

	try:
		# Get remaining data from rawpy
		raw = rawpy.imread(files[0])
		data['raw_format'] = True
		data['h'], data['w'] = raw.postprocess(user_flip=0).shape[:2]
		data['black_level'] = np.array(raw.black_level_per_channel)

		data['saturation_point'] = raw.white_level*sat_percent
		# The below code leads to inconsistent images for videos and light fields
		'''
		# For some cameras, the provided white_level is incorrect
		# TODO: per-channel saturation point
		long_img = rawpy.imread(files[-1]).raw_image_visible
		data['saturation_point'] = min(long_img[::2,::2].max(), long_img[::2,1::2].max(),
									   long_img[1::2,::2].max(), long_img[1::2,1::2].max()
									   )*sat_percent
		'''

		assert raw.camera_whitebalance[1] == raw.camera_whitebalance[3] or raw.camera_whitebalance[3] == 0, \
			   'Cannot figure out camera white_balance values'
		data['white_balance'] = raw.camera_whitebalance[:3]
		# Guess the bit-depth
		longest_exposure = np.argmax(data['exp'] * data['gain'] * data['aperture'])
		long_img = rawpy.imread(files[longest_exposure]).raw_image
		bits = int(np.ceil(np.log2(long_img.max())))
		data['libraw_scale'] = lambda img: img/(2**bits-1)*(2**16-1)
	except rawpy._rawpy.LibRawFileUnsupportedError:
		data['raw_format'] = False
		longest_exposure = np.argmax(data['exp'] * data['gain'] * data['aperture'])
		img = io.imread(files[longest_exposure])
		data['dtype'] = img.dtype
		assert len(img.shape) == 2, 'Provided files should not be demosaiced'
		data['h'], data['w'] = img.shape
		if bits is None:
			if img.dtype == np.float32:
				assert img.max() <= 1, 'For FP images, values should be in [0,1]'
				bits = 1
			elif img.dtype == np.uint16:
				bits = 16
				data['saturation_point'] = 2**16 - 1
			elif img.dtype == np.uint8:
				bits = 8
				data['saturation_point'] = 2**8 - 1
		data['saturation_point'] = 2**bits - 1
		shortest_exposure = np.argmin(data['exp'] * data['gain'] * data['aperture'])
		img = io.imread(files[shortest_exposure])
		data['black_level'] = np.array([black_level]*4)
		if np.abs(img.min() - black_level) > data['saturation_point'] * 0.01:
			logger.warning(f'Using black level {black_level}. Double check this with camera docs.')
		data['libraw_scale'] = lambda img: img/(2**bits-1)*(2**16-1)

	data['color_space'] = color_space.lower()

	logger.info(f"Stack contains {data['N']} images of size: {data['h']}x{data['w']}")
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
	# return (raw < saturation_threshold).all(axis=-1)
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


def encode(im1, im2):
	lin_max = np.max((im1, im2))
	lin_min = np.max((np.min((im1, im2)), 1e-10))

	# Do not stretch or compress histogram too much
	if lin_max/lin_min > 10000: lin_min = lin_max/10000
	if lin_max/lin_min < 1000: lin_min = lin_max/1000

	enc1 = np.log(im1/lin_min + 1) / np.log(lin_max/lin_min + 1) * 255
	enc2 = np.log(im2/lin_min + 1) / np.log(lin_max/lin_min + 1) * 255

	return enc1.astype(np.uint8), enc2.astype(np.uint8)


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
	import cv2
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

	matched_kp1 = np.zeros((len(matches), 1, 2), dtype=np.float32)
	matched_kp2 = np.zeros((len(matches), 1, 2), dtype=np.float32)

	for i in range(len(matches)):
		matched_kp1[i] = kp[matches[i].queryIdx].pt
		matched_kp2[i] = kp_ref[matches[i].trainIdx].pt

	H, _ = cv2.findHomography(matched_kp1, matched_kp2, cv2.RANSAC, 1)


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
