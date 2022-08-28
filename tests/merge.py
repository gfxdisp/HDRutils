import HDRutils, logging, os, numpy as np
from tqdm import trange, tqdm

logging.basicConfig(level=logging.ERROR)

'''
root = '/home/pmh64/gfxdisp/hdr_images/linkoping_2020/097'
root = '/home/pmh64/gfxdisp/hdr_images/linkoping_2020/036'

files = [os.path.join(root, f) for f in sorted(os.listdir(root))]

align = False
# HDR = HDRutils.merge(files, estimate_exp=None, do_align=align)
# HDRutils.imwrite('merged_base.hdr', HDR)
# # HDR = HDRutils.merge(files, estimate_exp='cerman', do_align=align)
# # HDRutils.imwrite('merged_cerman.hdr', HDR)
# HDR = HDRutils.merge(files, outlier=None, do_align=align)
# HDRutils.imwrite('merged_def.hdr', HDR)
# HDR = HDRutils.merge(files, do_align=align)
# HDRutils.imwrite('merged_out.hdr', HDR)

HDR, t = HDRutils.merge(files, estimate_exp='quick', do_align=align, outlier=None)
print(t)
HDRutils.imwrite('merged_quick.hdr', HDR)
'''

def merge_scene(root, scene):
	scene_root = os.path.join(root, scene)
	files = [os.path.join(scene_root, f) for f in sorted(os.listdir(scene_root))][1:-1]

	HDR, old, new = HDRutils.merge(files, estimate_exp='quick', outlier=None)
	HDRutils.imwrite(os.path.join(root, 'merged_quick', scene + '.hdr'), HDR)
	return (new - old)/old


def main(dataset):
	if dataset == 'linkoping':
		root = '/home/pmh64/gfxdisp/hdr_images/linkoping_2020'
		blacklist = ['009', '026', '052', '063', '074', '097', '098', '099', '100', '101', '114',
					 '117', '191', '065', '107', '024', '064', '078', '127', '155', '157', '160',
					 '186', 'benchmark', 'public']
		outfile = 'ratios_linkoping.csv'
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
					 'ignore_ghost.txt', 'fix_names.sh']
		outfile = 'ratios_fairchild.csv'

	data = []
	for scene in tqdm(sorted(os.listdir(root))):
		if not scene.startswith('merged') and scene not in blacklist:
			d = merge_scene(root, scene)
			data.append(d)
	np.savetxt(outfile, data, delimiter=',')


def get_files_and_exp(root):
	files = [f'{root}/img_{i:03d}.png' for i in range(11)]

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
	for loc in trange(100):
		root = os.path.join(hdr4cv, s, l, f'{loc:03d}', 'gain_1')
		files, exp = get_files_and_exp(root)

		old, new = HDRutils.merge(files, demosaic_first=False, color_space='raw', exp=exp, gain=np.ones_like(exp), estimate_exp='quick', outlier=None)
		ratios = (old - new)/new
		data.append(ratios)
	np.savetxt('ratios_hdr4cv.csv', data, delimiter=',')

main_hdr4cv()

main('linkoping')

main('fairchild')
