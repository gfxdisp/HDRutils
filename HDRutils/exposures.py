import numpy as np
import logging
import HDRutils
from tqdm import trange

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lsmr

# np.set_printoptions(linewidth=np.inf)
# np.set_printoptions(precision=2)
# np.set_printoptions(suppress=True)

logger = logging.getLogger(__name__)

def estimate_exposures(imgs, exif_exp, metadata, method, noise_floor=16, percentile=10,
					   invert_gamma=False, cam=None, outlier=None, solver='wls',
					   num_msts=50):
	"""
	Exposure times may be inaccurate. Estimate the correct values by fitting a linear system.
	
	:imgs: Image stack
	:exif_exp: Exposure times read from image metadata
	:metadata: Internal camera metadata dictionary
	:method: Pick from ['cerman', 'mst', 'pairwise']
	:noise_floor: All pixels smaller than this will be ignored
	:percentile: Use a small percentage of the least noisy pixels for the estimation
	:invert_gamma: If the images are gamma correct invert to work with linear values
	:cam: Camera noise parameters for better estimation
	:outlier: Pick from [None, 'cerman', 'tiled']

	:return: Corrected exposure times
	"""
	assert method in ('mst', 'cerman', 'pairwise')
	assert solver in ('base', 'ols', 'wls')
	assert outlier in (None, 'cerman', 'tiled')
	num_exp = len(imgs)
	assert num_exp > 1, 'Files not found or are invalid'

	# Mask out saturated and noisy pixels
	black_frame = np.tile(metadata['black_level'].reshape(2, 2), (metadata['h']//2, metadata['w']//2))

	if imgs.ndim == 4:
		assert imgs.shape[-1] == 3 or imgs[...,3].all() == 0
		black_frame = np.ones_like(imgs[0]) * metadata['black_level'][:3][None,None]
	Y = np.maximum(imgs - black_frame, 1e-6)	# Add epsilon since we need log(Y)
	if invert_gamma:
		max_value = np.iinfo(metadata['dtype']).max
		Y = (Y / max_value)**(invert_gamma) * max_value

	# Use green channel at reduced resolution
	num_pix = int(np.ceil(metadata['h']/2))*int(np.ceil(metadata['w']/2))
	black_frame = black_frame[::2,1::2]
	Y = Y[:,::2,1::2]

	if method == 'cerman':
		'''
		L. Cerman and V. Hlavac, “Exposure time estimation for high dynamic range imaging with
		hand held camera” in Proc. of Computer Vision Winter Workshop, Czech Republic. 2006.
		'''
		from skimage.exposure import histogram, match_histograms
		rows, cols, m, W = np.zeros((4, 0))
		for i in range(num_exp - 1):
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
		# argmin = lambda init, lmbda: lsmr(diags(W) @ O, W * m)[0]
		argmin = lambda init, lmbda: np.linalg.lstsq((diags(W) @ O).todense(), W * m)[0]


	elif method in ('mst', 'pairwise'):
		# If noise parameters is provided, retrieve variances, else use simplified model
		if not cam:
			scaled_var = 1/Y
		else:
			if cam == 'test':
				cam = HDRutils.NormalNoise('Sony', 'ILCE-7R', 100, bits=14)
			scaled_var = np.stack([(cam.var(y)/y**2) for y in Y/(2**cam.bits - 1)])*(2**cam.bits - 1)

		num_tiles = 16 	# Use (num_tiles)x(num_tiles) tiles; clip last few rows and cols
		_, h, w = Y.shape
		h = num_tiles * (h//num_tiles)
		w = num_tiles * (w//num_tiles)
		Y = Y[:,:h,:w].reshape(num_exp, num_tiles, h//num_tiles, num_tiles, w//num_tiles).transpose((0,1,3,2,4)).astype(np.float32)
		scaled_var = scaled_var[:,:h,:w].reshape(num_exp, num_tiles, h//num_tiles, num_tiles, w//num_tiles).transpose(0,1,3,2,4).astype(np.float32)
		black_frame = black_frame[:h,:w].reshape(num_tiles, h//num_tiles, num_tiles, w//num_tiles).transpose(0,2,1,3).astype(np.float32)

		# Don't use saturated or noisy pixels
		Y[Y + black_frame >= metadata['saturation_point']] = -1
		Y[Y <= noise_floor] = -1

		Y = Y.reshape(num_exp, num_tiles*num_tiles, -1)
		scaled_var = scaled_var.reshape(num_exp, num_tiles*num_tiles, -1)

		# Determine how many MSTs to select
		# This will be constant for all exposures for a given tile
		thresholds = np.sort(Y[1:])[...,-num_msts]
		valid = np.logical_and(Y[1:] > thresholds[...,None], Y[:-1] > -1)
		num_selected = valid.sum(axis=-1).min(axis=0)

		# Skip frames that are mostly noisy
		skip = num_selected < num_msts*0.5
		valid = np.logical_and(valid, np.logical_not(skip)[None,:,None])

		# Stopped vectorized processing since each tile will have different contributions
		# to the final linear system W.O.e = W.m
		W, O, m = [], [], []
		for tt in range(num_tiles*num_tiles):
			if skip[tt]: continue
			if solver == 'base':
				O_tile = np.zeros((num_selected[tt]*(num_exp-1), num_exp - 1), dtype=np.float32)
			else:
				O_tile = np.zeros((num_selected[tt]*(num_exp-1), num_exp), dtype=np.float32)
			W_tile, m_tile = [], []
			for ee in range(num_exp-1):
				# Pick "num_selected[tt]" highest weights that are valid in exposure ee+1
				# Then identify longest exposure for each pixel location
				if solver in ('base', 'ols'):
					weights = Y[ee,tt,valid[ee,tt]] + Y[ee+1,tt,valid[ee,tt]]
				else:
					weights = 1/(scaled_var[ee,tt,valid[ee,tt]] + scaled_var[ee+1,tt,valid[ee,tt]])
					# weights = 1/(1/Y[ee,tt,valid[ee,tt]] + 1/Y[ee+1,tt,valid[ee,tt]])
				idx = np.argsort(weights)[-num_selected[tt]:]
				longest = np.zeros_like(idx)
				if method == 'pairwise':
					longest[:] = ee + 1
				if method == 'mst':
					for ff in range(num_exp-1, ee, -1):
						candidates = Y[ff,tt,valid[ee,tt]][idx]
						longest[np.logical_and(longest == 0, candidates > -1,)] = ff
						if (longest > 0).all(): break
				# Update the linear system
				if solver == 'base':
					O_tile[ee*num_selected[tt]:(ee+1)*num_selected[tt],ee] = 1
					m_tile.append(Y[:,tt,valid[ee,tt]][:,idx][longest,np.arange(num_selected[tt])] / Y[ee,tt,valid[ee,tt]][idx])
					W_tile = np.ones_like(m_tile)
				else:
					weights = 1/(scaled_var[ee,tt,valid[ee,tt]][idx] + scaled_var[:,tt,valid[ee,tt]][:,idx][longest,np.arange(num_selected[tt])])
					# weights = 1/(1/Y[ee,tt,valid[ee,tt]][idx] + 1/Y[:,tt,valid[ee,tt]][:,idx][longest,np.arange(num_selected[tt])])
					W_tile.append(weights)
					m_tile.append(np.log(Y[:,tt,valid[ee,tt]][:,idx][longest,np.arange(num_selected[tt])] / Y[ee,tt,valid[ee,tt]][idx]))
					O_tile[ee*num_selected[tt]:(ee+1)*num_selected[tt],ee] = -1
					O_tile[ee*num_selected[tt]:(ee+1)*num_selected[tt]][np.arange(num_selected[tt]),longest] = 1
			W_tile, m_tile = [np.concatenate(a) for a in (W_tile, m_tile)]
			if outlier == 'tiled':
				e_tile = np.linalg.lstsq(np.diag(np.sqrt(W_tile)) @ O_tile, np.sqrt(W_tile) * m_tile, rcond=None)[0]
				e_tile = np.exp(e_tile - e_tile.max()) * exif_exp.max()
				if (np.abs(e_tile - exif_exp)/exif_exp > 1).any():
					continue

			O.append(O_tile)
			W.append(W_tile)
			m.append(m_tile)

		if len(W) == 0:
			logger.error(f'Exposure estimation failed. One or more of the frames are underexposed.')
			return exif_exp, np.nan
		W, O, m = [np.concatenate(a) for a in (W, O, m)]
		if solver == 'ols': W[:] = 1

		argmin = lambda init, lmbda: np.linalg.lstsq(np.diag(np.sqrt(W)) @ O, np.sqrt(W) * m, rcond=None)[0]

	if outlier == 'cerman':
		err_prev = np.finfo(float).max
		t = trange(1000, leave=False)
		for i in t:
			exp = argmin(exif_exp, 10)
			err = (W*(O @ exp - m))**2
			selected = err < 3*err.mean()
			W, m, O = W[selected], m[selected], O[selected]
			if err.mean() < 1e-6 or err_prev - err.mean() < 1e-6:
				break
			err_prev = err.mean()
			t.set_description(f'loss={err.mean()}')
		logger.info(f'Outliers removed {i} times.')

	exp = argmin(exif_exp, 10)

	if method == 'cerman' or solver == 'base':
		exp = np.append(exp, exif_exp[-1])
		for e in range(num_exp - 2, -1, -1):
			exp[e] = exif_exp[e+1]/exp[e]
	elif method in ('mst', 'pairwise'):
		exp = np.exp(exp - exp.max()) * exif_exp.max()

	logger.info(f'Exposure times in EXIF: {exif_exp}, estimated exposures: {exp}.')
	reject = np.abs(exp - exif_exp)/exif_exp > 2
	exp[reject] = exif_exp[reject]
	if reject.any():
		logger.warning(f'Exposure estimation failed {reject}. Reverting back to EXIF data for these values.')
	return exp
