import os, logging, sys
sys.path.append('..')
import HDRutils, numpy as np

# logging.basicConfig(level=logging.INFO)

root = 'image_stack'
for perc in (90, 0):
    print(f'Using {100 - perc}% of the pixels')
    print('Without static noise:')
    files = [os.path.join(root, f) for f in os.listdir(root) \
             if f.startswith('noisy_no_static')]

    metadata = {'black_level': np.array([0,0,0,0], dtype=np.float32),
                'gain': np.array([1,1,1,1], dtype=np.float32),
                'aperture': np.array([1,1,1,1], dtype=np.float32),
                'exp': np.arange(0.5, 2.5, 0.5, dtype=np.float32),
                'raw_format': False, 'dtype': HDRutils.imread(files[0]).dtype}

    exp = HDRutils.estimate_exposures(files, metadata, percentile=perc, noise_floor=1, invert_gamma=2.2)

    print('With static noise   :', end=' ')
    files = [os.path.join(root, f) for f in os.listdir(root) \
             if f.startswith('noisy_static')]
    exp = HDRutils.estimate_exposures(files, metadata, percentile=perc, noise_floor=1, invert_gamma=2.2)
