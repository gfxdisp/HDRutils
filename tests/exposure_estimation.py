import HDRutils, logging, os, numpy as np
from tqdm import tqdm
from gfxdisp import metrics as m
import pandas as pd
import functools

from gfxdisp import pfs
v = pfs.pfs()

logging.basicConfig(level=logging.ERROR)

pu = m.PU(L_max=5000)
psnr = lambda x, y: 20*np.log10(pu.peak/np.sqrt((x-y)**2).mean())

def process_scene(scene, temp, root, write=False, mul=4000):
	res = []
	bits = 14
	for g in range(5):
		g = 2**g
		iso = str(100*g)
		exp_times = 10**np.arange(-2, 2, dtype='float')/g
		gain = np.array([g]*len(exp_times))
		camera_make, camera_model = 'Canon', 'PowerShot S100'
		camera = HDRutils.NormalNoise(camera_make, camera_model, iso, bits=bits)

		# Simulate
		file = os.path.join(root, scene)
		clean = HDRutils.imread(file)
		clean = (clean / np.median(clean) * 1e-3).clip(max=1)
		clean = clean / clean.max()
		for i,e in enumerate(exp_times):
			quant_img = camera.simulate(clean, e, black_level=500)
			# v.view(quant_img.astype(np.float32) - 500); input(quant_img.min())
			quant_img[::2,::2,1] = quant_img[::2,::2,0]
			quant_img[1::2,1::2,1] = quant_img[1::2,1::2,2]
			HDRutils.imwrite(f'{temp}/noisy_static{i}.png', quant_img[...,1])

		# Merge
		row = {'scene':scene, 'gain':g}
		files = [os.path.join(temp, f) for f in os.listdir(temp) \
				if f.startswith('noisy_static')]
		noisy_exp = exp_times + np.random.randn(len(exp_times))*exp_times*0.15
		noisy_exp[-1] = exp_times[-1]
		merge = functools.partial(HDRutils.merge, files, exp=noisy_exp, gain=gain, black_level=500, demosaic_first=False, color_space='raw', saturation_percent=1, bits=bits, do_align=False)

		HDR, _ = merge(estimate_exp=False)
		row['noisy'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100

		HDR, _ = merge(estimate_exp='cerman', outlier='cerman')
		row['cerman-out'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100

		HDR, img = merge(estimate_exp='pairwise', outlier=None)
		if (HDR == noisy_exp).all():
			continue
		row['pairwise'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100
		HDRutils.imwrite('pair.hdr', img)

		HDR, img = merge(estimate_exp='mst', outlier=None)
		if (HDR == noisy_exp).all():
			continue
		row['mst'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100
		HDRutils.imwrite('mst.hdr', img)
		HDR, _ = merge(estimate_exp='quick', outlier=None, solver='base')
		if (HDR == noisy_exp).all():
			continue
		row['base'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100

		return row
		res.append(row)

	return res


def main():
	temp = 'image_stack'
	root = '/home/pmh64/gfxdisp/hdr_images/linkoping_2020/benchmark/sihdr/reference'
	root = '/home/pmh64/gfxdisp/hdr_images/Fairchild_Photo_Survey_Raws/merged'
	all_rows = []
	for scene in tqdm(sorted(os.listdir(root))):
		if scene.endswith(('.exr', '.hdr')):
			r = process_scene(scene, temp, root)
			all_rows.extend(r)

	df = pd.DataFrame(all_rows)
	df.to_csv('results.csv', index=False)

np.random.seed(0)
# main()
r = process_scene('507.hdr', 'image_stack', '/home/pmh64/gfxdisp/hdr_images/Fairchild_Photo_Survey_Raws/merged')
print(r)
