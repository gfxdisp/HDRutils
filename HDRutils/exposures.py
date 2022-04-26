import numpy as np, logging, HDRutils, gc
from tqdm import trange

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lsmr

import matplotlib.pyplot as plt
from gfxdisp import pfs

logger = logging.getLogger(__name__)
viewer = pfs.pfs()

def estimate_exposures(imgs, exif_exp, metadata, method, noise_floor=1, percentile=10,
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
	assert method in ('gfxdisp', 'cerman', 'batched_mst')
	assert outlier in (None, 'ransac', 'cerman')
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
	num_pix = int(np.ceil(metadata['h']/4))*int(np.ceil(metadata['w']/4))
	black_frame = black_frame[::4,1::4]
	Y = (Y[:,::4,1::4] + Y[:,1::4,::4])/2

	if method == 'cerman':
		'''
		L. Cerman and V. Hlavac, “Exposure time estimation for high dynamic range imaging with
		hand held camera” in Proc. of Computer Vision Winter Workshop, Czech Republic. 2006.
		'''
		from skimage.exposure import histogram, match_histograms
		rows, cols, m, W = np.zeros((4, 0))
		for i in range(num_exp - 1):
			# Ensure images are sorted in increasing order of exposure time
			# assert all(e1 <= e2 for e1, e2 in zip(exif_exp[:-1], exif_exp[1:])), \
			# 	   'Please name the input files in increasing order of exposure time when sorted'
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
		exp = lsmr(diags(W) @ O, W * m)[0]

	elif method in ('gfxdisp', 'batched_mst'):
		logger.info(f'Estimate using logarithmic linear system with noise model')		

		# If noise parameters is provided, retrieve variances, else use simplified model
		L = np.log(Y)
		if not cam:
			scaled_var = 1/Y
		else:
			if cam == 'test':
				cam = HDRutils.NormalNoise('Sony', 'ILCE-7R', 100, bits=14)
			scaled_var = np.stack([(cam.var(y)/y**2) for y in Y/(2**cam.bits - 1)])*(2**cam.bits - 1)

		# Construct logarithmic sparse linear system W.O.e = W.m
		logger.info(f'Constructing sparse matrix (O) and vector (m) using {num_pix} pixels')
		rows = np.arange(0, (num_exp - 1)*num_exp/2*num_pix, 0.5)
		cols, data = np.repeat(np.ones_like(rows)[None], 2, axis=0)
		data[1::2] = -1
		m, W = np.zeros((2, (num_exp - 1)*num_exp//2*num_pix), dtype=np.float32)
		cnt = 0
		for i in range(num_exp - 1):
			# Collect unsaturated pixels from all longer exposures
			for j in range(i + 1, num_exp):
				# Pick valid pixels with highest weights
				mask = np.stack((Y[i] + black_frame < metadata['saturation_point'],
								 Y[j] + black_frame < metadata['saturation_point'],
								 Y[i] > noise_floor, Y[j] > noise_floor)).all(axis=0)
				weights = np.sqrt(1/(scaled_var[i] + scaled_var[j]) * mask).flatten()
				if outlier in (('mst', 'batched_mst')):
					selected = np.arange(num_pix)		# Needed for MST to preserve order
				else:
					selected = np.argsort(weights)[-num_pix:]
				W[cnt*num_pix:(cnt + 1)*num_pix] = weights[selected]
				m[cnt*num_pix:(cnt + 1)*num_pix] = (L[i] - L[j]).flatten()[selected]
				cols[cnt*num_pix*2:(cnt + 1)*num_pix*2:2] = i
				cols[cnt*num_pix*2 + 1:(cnt + 1)*num_pix*2:2] = j
				cnt += 1

		O = csr_matrix((data, (rows, cols)), shape=((num_exp - 1)*num_exp//2*num_pix, num_exp))

		if method == 'batched_mst':
			cnt = 0
			edges = np.zeros((num_pix, (num_exp - 1)*num_exp//2, 3))
			for i in range(num_exp - 1):
				for j in range(i + 1, num_exp):
					edges[:,cnt,0] = W[cnt*num_pix:(cnt + 1)*num_pix]
					edges[:,cnt,1] = i
					edges[:,cnt,2] = j
					cnt += 1
			msts = kruskals_batched(edges, num_exp)
			x, y = np.triu_indices(num_exp, k=1)
			xx = x[None].repeat(num_pix, axis=0)
			yy = y[None].repeat(num_pix, axis=0)
			ww = np.zeros_like(xx)
			ww[:] = np.arange(num_pix)[:,None]
			mst_weights = np.convolve(edges[msts[ww, xx, yy],0], np.ones(num_exp), mode='valid')[::num_exp]
			selected = msts[ww, xx, yy].transpose(1,0).flatten()
			W, m, O = W[selected], m[selected], O[selected]
			# selected = mst_weights != 0
			# mst_weights = mst_weights[selected]
			# selected = np.concatenate([selected]*num_exp)
			# W, m, O = W[selected], m[selected], O[selected]

			# idx = np.argsort(mst_weights)[-num_pix//10:]
			# selected = np.concatenate([i*len(mst_weights) + idx for i in range(num_exp)])
			# W, m, O = W[selected], m[selected], O[selected]
			selected = W != 0
			W, m, O = W[selected], m[selected], O[selected]
			exp = lsmr(diags(W) @ O, W * m, x0=np.log(exif_exp), damp=1)[0]
		else:
			exp = lsmr(diags(W) @ O, W * m)[0]

	if outlier == 'cerman':
		err_prev = np.finfo(float).max
		t = trange(1000, leave=False)
		for i in t:
			exp = lsmr(diags(W) @ O, W * m)[0]
			err = (W*(O @ exp - m))**2
			selected = err < 3*err.mean()
			W, m, O = W[selected], m[selected], O[selected]
			if err.mean() < 1e-6 or err_prev - err.mean() < 1e-6:
				break
			err_prev = err.mean()
			t.set_description(f'loss={err.mean()}')
			del err; gc.collect()
		exp = lsmr(diags(W) @ O, W * m)[0]

	elif outlier == 'ransac':
		assert method == 'gfxdisp'
		num_rows = W.shape[0]
		# Randomly select 10% of the data
		selected = np.zeros(num_rows, dtype=bool)
		selected[:num_rows//10] = True
		loss = np.finfo(float).max
		WO = diags(W) @ O
		Wm = W*m
		t = trange(1000, leave=False)
		for i in t:
			np.random.shuffle(selected)
			exp_i = lsmr(WO[selected], Wm[selected])[0]
			exp_i = np.exp(exp_i - exp_i.max()) * exif_exp.max()
			reject = np.maximum(exp_i/exif_exp, exif_exp/exp_i) > 3
			exp_i[reject] = exif_exp[reject]
			err = ((W*(O @ exp_i - m))**2).sum()
			if err < loss:
				loss = err
				exp = np.log(exp_i)
				t.set_description(f'loss={err}; i={i}')
		exp = lsmr(diags(W) @ O, W * m)[0]

	if method == 'cerman':
		exp = np.append(exp, exif_exp[-1])
		for e in range(num_exp - 2, -1, -1):
			exp[e] = exif_exp[e+1]/exp[e]
	elif method in ('gfxdisp', 'batched_mst'):
		exp = np.exp(exp - exp.max()) * exif_exp.max()

	logger.info(f'Exposure times in EXIF: {exif_exp}, estimated exposures: {exp}. Outliers removed {i} times.')
	reject = np.maximum(exp/exif_exp, exif_exp/exp) > 3
	exp[reject] = exif_exp[reject]
	if reject.any():
		logger.warning(f'Exposure estimation failed {reject}. Reverting back to EXIF data for these values.')
	gc.collect()
	return exp


# Reference: https://github.com/choidami/sst/blob/main/core/kruskals/kruskals.py
def get_root(parents, node):
    # find path of objects leading to the root
    path = [node]
    root = parents[node]
    while root != path[-1]:
      path.append(root)
      root = parents[root]

    # compress the path and return
    for ancestor in path:
      parents[ancestor] = root
    return parents, root

def kruskals(weights_and_edges, n):
	sorted_edges = weights_and_edges[np.argsort(weights_and_edges[:,0]),1:][::-1]

	parents = np.arange(n)
	weights = np.ones(n)
	adj_matrix = np.zeros((n, n), dtype=bool)
	for i, j in sorted_edges:
		i, j = int(i), int(j)
		parents, root_i = get_root(parents, i)
		parents, root_j = get_root(parents, j)

		if root_i != root_j:
			# Combine two forests if i and j are not in the same forest.
			heavier = max([(weights[root_i], root_i), (weights[root_j], root_j)])[1]
			for r in [root_i, root_j]:
				if r != heavier:
					weights[heavier] = weights[heavier] + weights[r]
					parents[r] = heavier

			# Update top-right of adjacency matrix.
			adj_matrix[i][j] = True
	return adj_matrix


def get_root_batched(parents, node, n):
	bs = parents.shape[0]
	arange = np.arange(bs)
	# Find path of nodes leading to the root.
	path = np.zeros_like(parents)
	path[:, 0] = node
	root = parents[np.arange(bs), node]
	for i in range(1, n):
		path[:, i] = root
		root = parents[np.arange(bs), root]
	# Compress the path and return.
	for i in range(1, n):
		parents[arange, path[:, i]] = root
	return parents, root

def gather_numpy(self, dim, index):
    """
    Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    :param dim: The axis along which to index
    :param index: A tensor of indices of elements to gather
    :return: tensor of gathered values
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)

def kruskals_batched(weights_and_edges, n):
	"""Batched kruskal's algorithm for Maximumim spanning tree.
	Args:
		weights_and_edges: Shape (batch size, n * (n - 1) / 2, 3), where
			weights_and_edges[.][i] = [weight_i, node1_i, node2_i] for edge i.
		n: Number of nodes.
	Returns:
		Adjacency matrix. Shape (batch size, n, n)
	"""
	batch_size = weights_and_edges.shape[0]
	arange = np.arange(batch_size)
	# Sort edges based on weights, in descending order.
	sorted_weights = np.argsort(-1*weights_and_edges[:,:,0])
	dummy = sorted_weights[...,None].repeat(3, axis=-1)
	sorted_edges = gather_numpy(weights_and_edges, 1, dummy)[..., 1:]
	sorted_edges = sorted_edges.transpose((1,0,2))

	# Initialize weights and edges.
	weights = np.ones((batch_size, n))
	parents = np.arange(n)[None].repeat(batch_size, 0)

	adj_matrix = np.zeros((batch_size, n, n), dtype=bool)
	for edge in sorted_edges:
		i, j = edge[:,0].astype(int), edge[:,1].astype(int)
		parents, root_i = get_root_batched(parents, i, n)
		parents, root_j = get_root_batched(parents, j, n)
		is_i_and_j_not_in_same_forest = (root_i != root_j).astype(np.int32)

		# Combine two forests if i and j are not in the same forest.
		is_i_heavier_than_j = (
			weights[arange, root_i] > weights[arange, root_j]).astype(np.int32)
		weights_root_i = weights[arange, root_i] + (
			(weights[arange, root_j] * is_i_heavier_than_j)
			* is_i_and_j_not_in_same_forest +
			0.0 * (1.0 - is_i_and_j_not_in_same_forest))
		parents_root_i = (
			(root_i * is_i_heavier_than_j +  root_j * (1 - is_i_heavier_than_j)) 
			* is_i_and_j_not_in_same_forest +
			root_i * (1 - is_i_and_j_not_in_same_forest))
		weights_root_j = weights[arange, root_j] + (
			weights[arange, root_i] * (1 - is_i_heavier_than_j) 
			* is_i_and_j_not_in_same_forest +
			0.0 * (1.0 - is_i_and_j_not_in_same_forest))
		parents_root_j = (
			(root_i * is_i_heavier_than_j +  root_j * (1 - is_i_heavier_than_j)) 
			* is_i_and_j_not_in_same_forest +
			root_j * (1 - is_i_and_j_not_in_same_forest))
		weights[arange, root_i] = weights_root_i
		weights[arange, root_j] = weights_root_j
		parents[arange, root_i] = parents_root_i
		parents[arange, root_j] = parents_root_j

		# Update adjacency matrix.
		adj_matrix[arange, i, j] = is_i_and_j_not_in_same_forest
	return adj_matrix


def main():
	cost = np.array([[100, 2, 100, 6, 100],
		[2, 100, 3, 8, 5],
		[100, 3, 100, 100, 7],
		[6, 8, 100, 100, 9],
		[100, 5, 7, 9, 100]])
	num_nodes = cost.shape[0]
	edges = []
	for i in range(num_nodes-1):
		for j in range(i+1, num_nodes):
			if cost[i,j] != 100:	# 100 = nan (missing edge)
				edges.append((cost[i,j], i, j))
	edges = np.stack(edges)
	mst = kruskals(edges, num_nodes)
	print(mst)
	print(cost[mst], cost[mst].sum())
	print(np.where(mst))

	all_edges = np.stack([edges]*10)
	all_edges[::2,:,0] *= -1
	msts = kruskals_batched(all_edges, num_nodes)
	print(msts[:2])

if __name__=='__main__':
    main()
