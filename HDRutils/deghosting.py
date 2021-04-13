import logging
import cv2, numpy as np

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
