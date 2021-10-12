import os, logging, sys
from exp import estimate_exposures
sys.path.append('..')
import HDRutils, numpy as np

np.set_printoptions(suppress=True)
# logging.basicConfig(level=logging.INFO)

root = 'image_stack'
root = '/home/pmh64/gfxdisp'
files = [os.path.join(root, f) for f in sorted(os.listdir(root)) \
            if f.startswith('noisy_static')]
files = [os.path.join(root, f) for f in sorted(os.listdir(root))][1:-3]

metadata = {'black_level': 100,
            'gain': np.array([1,1,1,1], dtype=np.float32),
            'aperture': np.array([1,1,1,1], dtype=np.float32),
            'exp': np.array([0.0625, 0.25, 1], dtype=np.float32),
            'raw_format': False, 'dtype': HDRutils.imread(files[0]).dtype}
metadata = HDRutils.get_metadata(files)

exp = HDRutils.estimate_exposures(files, metadata)
camera_make, camera_model, iso = 'Canon', 'PowerShot S100', 100
model = HDRutils.NormalNoise(camera_make, camera_model, iso, bits=16)
exp = HDRutils.estimate_exposures(files, metadata, cam=model)
