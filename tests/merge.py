import sys
sys.path.insert(0, '..')
import HDRutils, logging, os
from tqdm import tqdm

# logging.basicConfig(level=logging.INFO)

root = 'image_stack'
files = [os.path.join(root, f) for f in os.listdir(root) \
		 if f.startswith('noisy_static')]

root = '/home/pmh64/gfxdisp/hdr_images/linkoping_2020'
for scene in sorted(os.listdir(root)):
	if isfile(os.path.join(root, 'merged_exp_wls'), f'{scene}.hdr'):
		continue
	if scene.startswith('merged'):
		continue
	print(f'Scene {scene}')
	scene_root = os.path.join(root, scene)
	files = [os.path.join(scene_root, f) for f in sorted(os.listdir(scene_root))]
	# camera_make, camera_model, iso = 'Canon', 'EOS 5D Mark III', 100
	camera_make, camera_model, iso = 'Sony', 'ILCE-7R', 100
	camera = HDRutils.NormalNoise(camera_make, camera_model, iso, bits=14)

	HDR = HDRutils.merge(files, do_align=False, estimate_exp=False)
	HDRutils.imwrite(os.path.join(root, 'merged', f'{scene}.hdr'), HDR)

	HDR = HDRutils.merge(files, do_align=False, estimate_exp=True)
	HDRutils.imwrite(os.path.join(root, 'merged_exp_ols', f'{scene}.hdr'), HDR)

	HDR = HDRutils.merge(files, do_align=False, estimate_exp=True, cam=camera)
	HDRutils.imwrite(os.path.join(root, 'merged_exp_wls', f'{scene}.hdr'), HDR)


# root = '/home/pmh64/gfxdisp/hdr_images/hdr_SIGGRAPH2019/hdr_037'
# root = '/home/fz261/gfxdisp_rsb/chart/20210816/far/0_raw/focal_stack_0/light_preset_-1'
# root = 'image_stack/gradient_whiteboard/0_raw/focal_stack_0/light_preset_-1'
# root = '/home/pmh64/gfxdisp/hdr_images/linkoping_2020/002'

# files = [os.path.join(root, f) for f in sorted(os.listdir(root))]
# camera_make, camera_model, iso = 'Sony', 'ILCE-7R', 100
# camera = HDRutils.NormalNoise(camera_make, camera_model, iso, bits=14)

# HDR = HDRutils.merge(files, do_align=False, estimate_exp=False)
# HDRutils.imwrite('merged.hdr', HDR)

# HDR = HDRutils.merge(files, do_align=False, estimate_exp=True)
# HDRutils.imwrite('merged_exp_ols.hdr', HDR)

# HDR = HDRutils.merge(files, do_align=False, estimate_exp=True, cam=camera)
# HDRutils.imwrite('merged_exp_wls.hdr', HDR)

# HDR = HDRutils.merge(files, do_align=False, estimate_exp=True, cam=camera)
# HDRutils.imwrite('merged_exp_pair.hdr', HDR)
