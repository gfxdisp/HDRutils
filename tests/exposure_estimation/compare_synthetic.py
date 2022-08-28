import HDRutils, logging, os, numpy as np
from tqdm import tqdm
from gfxdisp import metrics as m
import pandas as pd

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
		clean_pu = pu.encode(clean*mul)
		for i,e in enumerate(exp_times):
			quant_img = camera.simulate(clean, e, black_level=500)
			# v.view(quant_img.astype(np.float32) - 500); input(quant_img.min())
			quant_img[::2,::2,1] = quant_img[::2,::2,0]
			quant_img[1::2,1::2,1] = quant_img[1::2,1::2,2]
			HDRutils.imwrite(f'{temp}/noisy_static{i}.png', quant_img[...,1])

		# Merge
		row = {'scene':scene.strip('.exr'), 'gain':g}
		files = [os.path.join(temp, f) for f in os.listdir(temp) \
				if f.startswith('noisy_static')]
		noisy_exp = exp_times + np.random.randn(len(exp_times))*exp_times*0.15
		noisy_exp[-1] = exp_times[-1]

		HDR, _ = HDRutils.merge(files, exp=exp_times, gain=gain, black_level=500, estimate_exp=False, demosaic_first=False, color_space='raw', saturation_percent=1, bits=bits)
		# HDR = HDR/(2**camera.bits - 1)
		# row['base'] = psnr(pu.encode(HDR*mul), clean_pu)
		row['base'] = 0

		HDR, _ = HDRutils.merge(files, exp=noisy_exp, gain=gain, black_level=500, estimate_exp=False, demosaic_first=False, color_space='raw', saturation_percent=1, bits=bits)
		# HDR = HDR/(2**camera.bits - 1)
		# row['noisy'] = psnr(pu.encode(HDR*mul), clean_pu)
		row['noisy'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100

		HDR, row['time-cerman'] = HDRutils.merge(files, exp=noisy_exp, gain=gain, black_level=500, demosaic_first=False, color_space='raw', do_align=False, estimate_exp='cerman', saturation_percent=1, bits=bits)
		# HDR = HDR/(2**camera.bits - 1)
		# row['cerman'] = psnr(pu.encode(HDR*mul), clean_pu)
		row['cerman'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100

		HDR, row['time-mst'] = HDRutils.merge(files, exp=noisy_exp, gain=gain, black_level=500, demosaic_first=False, color_space='raw', estimate_exp='batched_mst', saturation_percent=1, outlier=None, cam=camera, bits=bits)
		# HDR = HDR/(2**camera.bits - 1)
		# row['mst'] = psnr(pu.encode(HDR*mul), clean_pu)
		row['mst'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100

		HDR, row['time-mst-out'] = HDRutils.merge(files, exp=noisy_exp, gain=gain, black_level=500, demosaic_first=False, color_space='raw', estimate_exp='batched_mst', saturation_percent=1, cam=camera, bits=bits)
		# HDR = HDR/(2**camera.bits - 1)
		# row['mst_cam'] = psnr(pu.encode(HDR*mul), clean_pu)
		row['mst_out'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100

		HDR, row['time-quick'] = HDRutils.merge(files, exp=noisy_exp, gain=gain, black_level=500, demosaic_first=False, color_space='raw', estimate_exp='quick', saturation_percent=1, outlier=None, bits=bits)
		# HDR = HDR/(2**camera.bits - 1)
		# row['quick'] = psnr(pu.encode(HDR*mul), clean_pu)
		row['quick'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100

		HDR, row['time-quick-out'] = HDRutils.merge(files, exp=noisy_exp, gain=gain, black_level=500, demosaic_first=False, color_space='raw', estimate_exp='quick', saturation_percent=1, bits=bits)
		# HDR = HDR/(2**camera.bits - 1)
		# row['quick_cam'] = psnr(pu.encode(HDR*mul), clean_pu)
		row['quick_out'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100

		HDR, row['time-quick-cam'] = HDRutils.merge(files, exp=noisy_exp, gain=gain, black_level=500, demosaic_first=False, color_space='raw', estimate_exp='quick', cam=camera, saturation_percent=1, outlier=None, bits=bits)
		# HDR = HDR/(2**camera.bits - 1)
		# row['quick_cam'] = psnr(pu.encode(HDR*mul), clean_pu)
		row['quick_cam'] = np.sqrt((((HDR-exp_times)/exp_times)[:-1]**2).mean())*100

		res.append(row)

		print(row['noisy'], row['cerman'], row['mst'], row['mst_out'], row['quick'], row['quick_out'], row['quick_cam'])
	return res


def main()
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

main()
