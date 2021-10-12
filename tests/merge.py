import sys
sys.path.append('..')
import HDRutils, logging, os
from tqdm import tqdm

# logging.basicConfig(level=logging.INFO)

root = 'image_stack'
files = [os.path.join(root, f) for f in os.listdir(root) \
		 if f.startswith('noisy_static')]

root = '/home/pmh64/gfxdisp/hdr_images/linkoping_2020/008'
files = [os.path.join(root, f) for f in sorted(os.listdir(root))][1:-2]
camera_make, camera_model, iso = 'Canon', 'EOS 5D Mark III', 100
camera = HDRutils.NormalNoise(camera_make, camera_model, iso, bits=14)

HDR = HDRutils.merge(files, do_align=False, estimate_exp=False)
HDRutils.imwrite('none.hdr', HDR)

HDR = HDRutils.merge(files, do_align=False, estimate_exp=True)
HDRutils.imwrite('naive.hdr', HDR)

HDR = HDRutils.merge(files, do_align=False, estimate_exp=True, cam=camera)
HDRutils.imwrite('opt.hdr', HDR)
