import HDRutils, logging, os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

cs = 'sRGB'
root = 'image_stack'
files = [os.path.join(root, f) for f in os.listdir(root)][:-1]

HDR = HDRutils.merge(files, do_align=True)
HDRutils.imwrite('merged.hdr', HDR)
