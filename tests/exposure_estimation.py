import os, logging, sys
sys.path.insert(0, '..')
import HDRutils, numpy as np

np.random.seed(0)
np.set_printoptions(suppress=True)
# logging.basicConfig(level=logging.INFO)

file = 'image_stack/clean_sky.jpg'
clean = HDRutils.imread(file)
clean = (clean / 255)**(2.2)
# G = np.logspace(np.log10(2**(-8)), 0 , 1000)
# clean = np.stack([(np.stack([G]*1000))]*3, axis=-1)

camera_make, camera_model = 'Canon', 'PowerShot S100'
exp_times, iso = 2**np.arange(-4, 1, 2, dtype=float)/32, '3200'
model = HDRutils.NormalNoise(camera_make, camera_model, iso)
imgs = np.stack([model.simulate(clean, e, bits=14, black_level=100)[...,1] \
                 for e in exp_times]).astype(np.float32)

metadata = {'black_level': 100,
            'saturation_point': 2**14 - 1,
            'gain': np.array([1,1,1,1], dtype=np.float32),
            'aperture': np.array([1,1,1,1], dtype=np.float32),
            'exp': np.array([0.0625, 0.25, 1], dtype=np.float32),
            'raw_format': False, 'dtype': imgs.dtype}

# root = '/home/pmh64/gfxdisp/hdr_images/linkoping_2020/008'
# files = [os.path.join(root, f) for f in sorted(os.listdir(root))][2:-2]
# metadata = HDRutils.get_metadata(files)
# imgs = np.stack([HDRutils.imread(f).astype(np.float32)[...,1] for f in files])

metadata['N'], metadata['h'], metadata['w'] = imgs.shape
print('Gfxdisp')
exp = HDRutils.estimate_exposures(imgs, metadata['exp'], metadata, 'l2', 10, method='gfxdisp')
print('cerman')
exp = HDRutils.estimate_exposures(imgs, metadata['exp'], metadata, 'l2', 10, method='cerman')
# camera_make, camera_model, iso = 'Canon', 'EOS 5D Mark III', 100
# model = HDRutils.NormalNoise(camera_make, camera_model, iso, bits=16)
# exp = HDRutils.estimate_exposures(files, metadata, cam=model)
