import logging, sys
sys.path.insert(0, '..')
import HDRutils

logging.basicConfig(level=logging.INFO)

img = HDRutils.imread('/home/pmh64/gfxdisp/hdr_images/linkoping_2020/merged_itmo/reconstructions/expandnet/clip_95/015.exr')
ref = HDRutils.imread('/home/pmh64/gfxdisp/hdr_images/linkoping_2020/merged_itmo/hdr/015.exr')

HDRutils.scatter_pixels(ref, img)
