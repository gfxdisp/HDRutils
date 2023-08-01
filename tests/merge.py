import HDRutils, logging, os, numpy as np
from tqdm import trange, tqdm
import functools

# logging.basicConfig(level=logging.ERROR)
from gfxdisp import pfs
viewer = pfs.pfs()


# '''
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
root = '/home/pmh64/gfxdisp/hdr_images/linkoping_2020/008'
# root = '/home/pmh64/gfxdisp/pmh64/slant_edge/hdr/'
# root = '/home/pmh64/gfxdisp/hdr_images/Fairchild_Photo_Survey_Raws/Air_Bellows_Gap'

files = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.lower().endswith('.cr2')]

align = True
merge = functools.partial(HDRutils.merge, files, do_align=align, return_exif_exp=True)
HDR, old, new = merge()
HDRutils.imwrite('merged_exif.hdr', HDR)
HDR, old, new = merge(estimate_exp='mst')
print((old - new)/old)
HDRutils.imwrite('merged_mst.hdr', HDR)
HDR, old, new = merge(estimate_exp='cerman', outlier='cerman')
print((old - new)/old)
HDRutils.imwrite('merged_cerman.hdr', HDR)
HDR, old, new = merge(estimate_exp='mst', solver='base')
print((old - new)/old)
HDRutils.imwrite('merged_baseline.hdr', HDR)
# '''

def merge_scene(root, scene, method, out):
	scene_root = os.path.join(root, scene)
	files = [os.path.join(scene_root, f) for f in sorted(os.listdir(scene_root))][1:-1]

	HDR, old, new = HDRutils.merge(files, estimate_exp=method, outlier=out)
	outfile = f'merged_{method}' if out is None else f'merged_{method}_out'
	HDRutils.imwrite(os.path.join(root, outfile, scene + '.hdr'), HDR)
	return (old - new)/old


def main(dataset, method='mst', out=None):
	if dataset == 'linkoping':
		root = '/home/pmh64/gfxdisp/hdr_images/linkoping_2020'
		blacklist = ['009', '026', '052', '063', '074', '097', '098', '099', '100', '101', '114',
					 '117', '191', '065', '107', '024', '064', '078', '127', '155', '157', '160',
					 '186', 'benchmark', 'public']
		outfile = 'ratios_linkoping.csv' if out is None else 'ratios_linkoping_out.csv'
	elif dataset == 'fairchild':
		root = '/home/pmh64/gfxdisp/hdr_images/Fairchild_Photo_Survey_Raws'
		blacklist = ['comparison', 'Tupper_Lake_1', 'Hall_of_Fame', 'Bar_Harbor_Presunrise',
					 'Ben&Jerry\'s', 'Canadian_Falls', 'Devil\'s_Bathtub', 'Exploratorium_1',
					 'Exploratorium_1', 'Frontier', 'Golden_Gate_1', 'Golden_Gate_2',
					 'Half_Dome_Sunset', 'Little_River', 'Mackinac_Bridge', 'Mammoth_Hot_Springs',
					 'Mason_Lake_2', 'McKee\'s_Pub', 'Middle_Pond', 'Mirror_Lake',
					 'Niagara_Falls', 'Otter_Point', 'Peck_Lake', 'Peppermill', 'Redwood_Sunset',
					 'Road\'s_End_Fire_Damage', 'Round_Barn_Inside', 'South_Branch_Kings_River',
					 'The_Grotto', 'Tupper_Lake_2', 'Waffle_House', 'West_Branch_Ausable_1',
					 'West_Branch_Ausable_2', 'Letchworth_Tea_Table_2', 'Luxo_Double_Checker',
					 'Flamingo', 'Ahwahnee_Great_Lounge', 'ignore_ghost.txt', 'fix_names.sh']
		outfile = 'ratios_fairchild.csv' if out is None else 'ratios_fairchild_out.csv'

	data = []
	for scene in tqdm(sorted(os.listdir(root))):
		if not scene.startswith('merged') and scene not in blacklist:
			d = merge_scene(root, scene, method, out)
			data.append(d)
	np.savetxt(outfile, data, delimiter=',')


def get_files_and_exp(root):
	# files = [f'{root}/img_{i:03d}.png' for i in range(11)]
	files = [os.path.join(root, f) for f in os.listdir(root) if f[7:] == '.png']

	exp = []
	with open(os.path.join(root, 'exposures.txt'), 'r') as file:
		for line in file.readlines():
			line = line.split(', ')
			if len(line) == 1:
				line = line[0].split(' ')
			exp.append(1/float(line[1]))
			if len(exp) == len(files):
				break
	return files, exp

def main_hdr4cv():
	hdr4cv = '/anfs/gfxdisp/hdr4cv'
	s, l = 'Faces', 'optimal'
	data = []
	for s in tqdm(('Faces', 'Tunnel', 'Street-parallel', 'Street-diagonal')):
		for l in ('Glare', 'HDR', 'Night', 'optimal'):
			for loc in trange(100, leave=False):
				root = os.path.join(hdr4cv, s, l, f'{loc:03d}', 'gain_1')
				files, exp = get_files_and_exp(root)
				if len(files) == 0: continue

				HDR, old, new = HDRutils.merge(files, demosaic_first=False, color_space='raw', exp=exp, gain=np.ones_like(exp), estimate_exp='cerman', outlier='cerman')
				ratios = (old - new)/old
				data.append(ratios)
	np.savetxt('ratios_hdr4cv.csv', np.array(data), delimiter=',', fmt='%s')

# main_hdr4cv()

# main('linkoping')
# main('linkoping', out='tiled')

# main('fairchild')
# main('fairchild', out='tiled')
