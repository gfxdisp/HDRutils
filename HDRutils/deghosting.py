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

def align(imgs, ref):
	"""
	Align the images. Use feature matching and homography estimation to align
	input images. This works well for camera motion when scene depth is small.

	:imgs: input image stack
	:ref: index of reference image
	:return: aligned images
	"""
	logger = logging.getLogger('align')
	logger.info(f"Aligning images using homography. Reference provided: {ref}")
	h, w = imgs[ref].shape[:2]
	aligned = []

	logger.info('Using SIFT feature detector')
	try:
		detector = cv2.xfeatures2d.SIFT_create()
	except:
		detector = cv2.SIFT_create()
	bf = cv2.BFMatcher(crossCheck=True)

	for i, img in enumerate(imgs):
		if i == ref:
			aligned.append(img)
		else:
			im1, im2 = encode(imgs[ref], img)
			kp_ref, desc_ref = detector.detectAndCompute(im1, None)
			kp, desc = detector.detectAndCompute(im2, None)

			matches = bf.match(desc, desc_ref)
			logger.info(f'{len(matches)} matches found for image {i}')
			matches = sorted(matches, key=lambda x:x.distance)[:100]

			# img = cv2.drawMatches((img//256).astype(np.uint8), kp, (imgs[ref]//256).astype(np.uint8), kp_ref, matches, None)

			H = find_homography(kp, kp_ref, matches)
			logger.info(f'Estimated homography: {H}')
			warped = cv2.warpPerspective(img, H, (w, h))

			aligned.append(warped)

	return aligned
