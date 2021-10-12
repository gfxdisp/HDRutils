import os, logging, sys
sys.path.append('..')
import HDRutils, numpy as np

# logging.basicConfig(level=logging.INFO)

file = 'image_stack/clean_forest.jpg'
clean = HDRutils.imread(file)
clean = (clean / 255)**(2.2)
# G = np.logspace(np.log10(2**(-8)), 0 , 1000)
# clean = np.stack([(np.stack([G]*1000))]*3, axis=-1)

camera_make, camera_model = 'Canon', 'PowerShot S100'
exp_times, iso = 2**np.arange(-4, 1, 2, dtype=float), 100
model = HDRutils.NormalNoise()

for i,e in enumerate(exp_times):
    quant_img = model.simulate(clean, camera_make, camera_model, e, iso, disable_static_noise=True, bits=16)
    # quant_img = model.simulate(clean*(2**14-1), e, iso, disable_static_noise=True)
    HDRutils.imwrite(f'image_stack/noisy_no_static{i}.png', quant_img)

    quant_img = model.simulate(clean, camera_make, camera_model, e, iso, disable_static_noise=False, bits=16)
    # quant_img = model.simulate(clean*(2**14-1), e, iso, disable_static_noise=False)
    HDRutils.imwrite(f'image_stack/noisy_static{i}.png', quant_img)
