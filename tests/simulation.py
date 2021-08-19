import os, logging, sys
sys.path.append('..')
import HDRutils, numpy as np

logging.basicConfig(level=logging.INFO)

file = 'image_stack/clean_forest.jpg'
clean = HDRutils.imread(file)
clean = (clean / 255)**(2.2) * 255
# G = np.logspace(np.log10(2**(-8)), 0 , 1000)
# clean = np.stack([(np.stack([G]*1000))]*3, axis=-1)

camera_make, camera_model = 'Canon', 'PowerShot S100'
exp_times, iso = np.arange(0.5, 2.5, 0.5) / 32, 3200
model = HDRutils.NormalNoise()

for i,e in enumerate(exp_times):
    noisy_img = model.simulate(clean, camera_make, camera_model, e, iso, disable_static_noise=True)
    noisy_img = np.maximum(noisy_img, 0)
    noisy_img = (noisy_img / 255)**(1/2.2) * 255
    quant_img = noisy_img.clip(0, 255).astype(np.uint8)
    HDRutils.imwrite(f'image_stack/noisy_no_static{i}.png', quant_img)

    noisy_img = model.simulate(clean, camera_make, camera_model, e, iso, disable_static_noise=False)
    noisy_img = np.maximum(noisy_img, 0)
    noisy_img = (noisy_img / 255)**(1/2.2) * 255
    quant_img = noisy_img.clip(0, 255).astype(np.uint8)
    HDRutils.imwrite(f'image_stack/noisy_static{i}.png', quant_img)
