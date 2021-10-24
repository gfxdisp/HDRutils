import os, logging, sys
sys.path.insert(0, '..')
import HDRutils, numpy as np

# logging.basicConfig(level=logging.INFO)

file = 'image_stack/clean_sky.jpg'
clean = HDRutils.imread(file)
clean = (clean / 255)**(2.2)
# G = np.logspace(np.log10(2**(-8)), 0 , 1000)
# clean = np.stack([(np.stack([G]*1000))]*3, axis=-1)

camera_make, camera_model = 'Canon', 'PowerShot S100'
exp_times, iso = 2**np.arange(-4, 1, 2, dtype=float), '100'
model = HDRutils.NormalNoise(camera_make, camera_model, iso)

for i,e in enumerate(exp_times):
    quant_img = model.simulate(clean, e, bits=16, black_level=100)
    HDRutils.imwrite(f'image_stack/noisy_static{i}.png', quant_img)
