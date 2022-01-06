import logging, os, tqdm, gc
from numpy.core.numeric import ones_like
from fractions import Fraction
import exifread

import numpy as np, cv2
from tqdm import trange
import rawpy

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lsqr

import HDRutils
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
	data['N'] = len(files)

	try:
		data['exp'], data['gain'], data['aperture'] = np.empty((3, data['N']))
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
		# For some cameras, the provided white_level is incorrect
		data['saturation_point'] = raw.white_level*sat_percent
		assert raw.camera_whitebalance[1] == raw.camera_whitebalance[3] or raw.camera_whitebalance[3] == 0, \
			   'Cannot figure out camera white_balance values'
		data['white_balance'] = raw.camera_whitebalance[:3]
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


def estimate_exposures(imgs, exif_exp, metadata, method, noise_floor=16, percentile=10,
					   invert_gamma=False, cam=None, outlier='cerman'):
	"""
	Exposure times may be inaccurate. Estimate the correct values by fitting a linear system.
	
	:imgs: Image stack
	:exif_exp: Exposure times read from image metadata
	:metadata: Internal camera metadata dictionary
	:method: Pick from ['gfxdisp', 'cerman']
	:noise_floor: All pixels smaller than this will be ignored
	:percentile: Use a small percentage of the least noisy pixels for the estimation
	:invert_gamma: If the images are gamma correct invert to work with linear values
	:cam: Camera noise parameters for better estimation
	:return: Corrected exposure times
	"""
	assert method in ('gfxdisp', 'cerman')
	num_exp = len(imgs)
	assert num_exp > 1, f'Files not found or are invalid: {files}'

	# Mask out saturated and noisy pixels
	black_frame = np.tile(metadata['black_level'].reshape(2, 2), (metadata['h']//2, metadata['w']//2)) \
				  if metadata['raw_format'] else metadata['black_level']

	Y = np.maximum(imgs - black_frame, 1e-6)	# Add epsilon since we need log(Y)
	if invert_gamma:
		max_value = np.iinfo(metadata['dtype']).max
		Y = (Y / max_value)**(invert_gamma) * max_value

	if method == 'cerman':
		'''
		L. Cerman and V. Hlavac, “Exposure time estimation for high dynamic range imaging with
		hand held camera” in Proc. of Computer Vision Winter Workshop, Czech Republic. 2006.
		'''
		from skimage.exposure import histogram, match_histograms
		rows, cols, m, W = np.zeros((4, 0))
		for i in range(num_exp - 1):
			# Ensure images are sorted in increasing order of exposure time
			assert all(e1 <= e2 for e1, e2 in zip(exif_exp[:-1], exif_exp[1:])), \
				   'Please name the input files in increasing order of exposure time when sorted'
			im1, im2 = Y[i], Y[i+1]
			mask = np.stack((im1 + black_frame < metadata['saturation_point'],
							 im2 + black_frame < metadata['saturation_point'],
							 im1 > noise_floor, im2 > noise_floor)).all(axis=0)

			# Match histograms of consecutive exposures
			im1_hat = match_histograms(im1, im2)
			im2_hat = match_histograms(im2, im1)

			# Construct the simple sparse linear system. There are 2 sets for each pair (Eq. 4)
			num_pix = np.count_nonzero(mask)
			rows = np.concatenate((rows, np.arange(2*num_pix) + len(rows)))
			cols = np.concatenate((cols, np.repeat(i, 2*num_pix)))
			m = np.concatenate((m, (im1_hat[mask]/im1[mask]), (im2[mask]/im2_hat[mask])))
			
			# Weights are given by sqrt() of histogram counts (Eq. 4)
			im1, im2 = im1.astype(np.uint16), im2.astype(np.uint16)
			counts, bins = histogram(im1)
			weights1 = np.sqrt(counts[np.searchsorted(bins, im1[mask])])
			counts, bins = histogram(im2)
			weights2 = np.sqrt(counts[np.searchsorted(bins, im2[mask])])
			W = np.concatenate((W, weights1, weights2))

		num_rows = rows.shape[0]
		data = np.ones(num_rows)
		O = csr_matrix((data, (rows, cols)), shape=(num_rows, (num_exp - 1)))

	elif method == 'gfxdisp':
		logger.info(f'Estimate using logarithmic linear system with noise model')
		num_pix = int(percentile/100*metadata['h']*metadata['w'])
		
		# If noise parameters is provided, retrieve variances, else use simplified model
		L = np.log(Y)
		if cam == 'default':
			cam = HDRutils.NormalNoise('Sony', 'ILCE-7R', 100, bits=14)
		bits = cam.bits if cam else 14
		scaled_var = np.stack([(cam.var(y)/y**2) if cam else 1/y**2 for y in Y/(2**bits - 1)])

		# Construct logarithmic sparse linear system W.O.e = W.m
		logger.info(f'Constructing sparse matrix (O) and vector (m) using {num_pix} pixels')
		rows = np.arange(0, (num_exp - 1)*num_pix, 0.5)
		cols, data = np.repeat(np.ones_like(rows)[None], 2, axis=0)
		data[1::2] = -1
		m = np.zeros((num_exp - 1)*num_pix, dtype=np.float32)
		W = np.zeros_like(m)
		for i in range(num_exp - 1):
			cols[i*num_pix*2:(i + 1)*num_pix*2:2] = i
			# Collect unsaturated pixels from all longer exposures
			for j in range(i + 1, num_exp):
				mask = np.stack((Y[i] + black_frame < metadata['saturation_point'],
								 Y[j] + black_frame < metadata['saturation_point'],
								 Y[i] > noise_floor, Y[j] > noise_floor)).all(axis=0)
				# if mask.sum() < num_pix:
				# 	continue
				weights = np.concatenate((W[i*num_pix:(i+1)*num_pix],
										 (1/(scaled_var[i] + scaled_var[j]) * mask).flatten()))
				logdiff = np.concatenate((m[i*num_pix:(i+1)*num_pix], (L[i] - L[j]).flatten()))
				selected = np.argsort(weights)[-num_pix:]
				W[i*num_pix:(i + 1)*num_pix] = weights[selected]
				m[i*num_pix:(i + 1)*num_pix] = logdiff[selected]
				cols[i*num_pix*2 + 1:(i + 1)*num_pix*2:2][selected > num_pix] = j

		O = csr_matrix((data, (rows, cols)), shape=((num_exp - 1)*num_pix, num_exp))

	logger.info('Solving the sparse linear system using least squares')
	if outlier == 'cerman':
		err_prev = np.finfo(float).max
		t = trange(1000, leave=False)
		for i in t:
			exp = lsqr(diags(W) @ O, W * m)[0]
			err = (W*(O @ exp - m))**2
			selected = err < 3*err.mean()
			W = W[selected]
			m = m[selected]
			O = O[selected]
			if err.mean() < 1e-6 or err_prev - err.mean() < 1e-6:
				# assert err_prev - err.mean() > 0
				break
			err_prev = err.mean()
			t.set_description(f'loss={err.mean()}')
			del err, selected
			gc.collect()
		logger.warning(f'Used {O.shape[0]/(num_exp - 1)/num_pix*100}% of the initial pixels')

	elif outlier == 'ransac':
		assert method == 'gfxdisp'
		num_rows = W.shape[0]
		# Randomly select 10% of the data
		selected = np.zeros(num_rows, dtype=bool)
		selected[:num_rows//10] = True
		loss = np.finfo(float).max
		WO = diags(W) @ O
		Wm = W*m
		t = trange(100, leave=False)
		for i in t:
			np.random.shuffle(selected)
			exp_i = lsqr(WO[selected], Wm[selected])[0]
			exp_i = np.exp(exp_i - exp_i.max()) * exif_exp.max()
			reject = np.maximum(exp_i/exif_exp, exif_exp/exp_i) > 3
			exp_i[reject] = exif_exp[reject]
			err = ((W*(O @ exp_i - m))**2).sum()
			if err < loss:
				loss = err
				exp = np.log(exp_i)
				t.set_description(f'loss={err}; i={i}')

	else:
		exp = lsqr(diags(W) @ O, W * m)[0]


	if method == 'cerman':
		exp = np.append(exp, exif_exp[-1])
		for e in range(num_exp - 2, -1, -1):
			exp[e] = exif_exp[e+1]/exp[e]
	elif method == 'gfxdisp':
		exp = np.exp(exp - exp.max()) * exif_exp.max()

	# logger.warning(f'Exposure times in EXIF: {exif_exp}, estimated exposures: {exp}. Outliers removed {i} times')
	# reject = np.maximum(exp/exif_exp, exif_exp/exp) > 3
	# exp[reject] = exif_exp[reject]
	# if reject.any():
	# 	logger.warning(f'Exposure estimation failed {reject}. Try using more pixels')
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
