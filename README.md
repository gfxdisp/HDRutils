# HDRutils

Some utility functions to generate HDR images from a sequence of exposure time or gain modulated images. You can find a separate README describing some functinos for realistic noise simulations [here](HDRutils/noise_modeling).

**Table of contents**
- [Installation](#installation)
    - [Additional dependencies](#additional-dependencies)
- [HDR image I/O](#reading-and-writing)
- [Capture](#multi-exposure-capture)
- [HDR merge](#merge-input-images)
    - [Demosaicing](#merge-and-demosaic-or-demosaic-and-merge)
    - [RAW bayer input](#merge-raw-bayer-frames-from-non-raw-formats)
    - [Alignment](#alignment)
    - [Exposure estimation](#exposure-estimation)
- [Noise simulation](#noise-simulation)
- [Citation](#citation)

## Installation
To download HDRUtils, use Pypi via pip:

    pip install HDRutils

If you prefer cloning this repository, install the dependencies using pip:
    
    pip clone https://github.com/gfxdisp/HDRutils.git
    cd HDRutils
    pip install -e .

### Additional dependencies
You will need the [FreeImage plugin](https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.freeimage.html) for reading and writing OpenEXR images:

    imageio_download_bin freeimage

If you wish to capture HDR stacks using a DSLR, you will need gphoto2:

    sudo apt install gphoto2


## Reading and writing
Simple wrapper functions for [imageio's](https://imageio.github.io/) `imread` and `imwrite` are provided to set appropriate flags for HDR data. You can even call `imread` on RAW file formats:

```python
import HDRutils

raw_file = 'example_raw.arw'
img_RGB = HDRutils.imread(raw_file)

hdr_file = 'example.exr'
img = HDRutils.imread(raw_file)

HDRutils.imwrite('rgb.png', img_RGB)
HDRutils.imwrite('output_filename.exr', img)
```

## Multi-exposure capture
Make sure gphoto2 is installed. Additionally, set camera to **manual mode** and **disable autofocus** on the lens. Then, decide valid exposure times (by scrolling on the camera) and run:

```python
from HDRutils.capture import DSLR
camera = DSLR(ext='.arw')
exposures = ['10', '1', '1/10', '1/100']
camera.capture_HDR_stack('image', exposures)
```

## Merge input images
The [rawpy](https://github.com/letmaik/rawpy) wrapper is used to read RAW images. [Noise-aware merging](https://www.cl.cam.ac.uk/research/rainbow/projects/noise-aware-merging/) is performed using the Poisson-noise optimal estimator. The generated HDR image is linearly related to the scene radiance

```python
files = ['`image_0.arw`', '`image_1.arw`', '`image_2.arw`']		# RAW input files
HDR_img = HDRutils.merge(files)[0]
HDRutils.imwrite('merged.exr', HDR_img)
```

Sometimes the shortest exposure may contain saturated pixels. These cause artifacts when manual white-balance/color calibration is performed. Thus, `HDRutils.merge()` returns an unsaturated mask in addition to the merged image. The saturated pixels can be clipped after manual white-balance/color calibration.

### Merge and demosaic or demosaic and merge?
The default function processes each image individually using [libraw](https://www.libraw.org/) and then merges the RGB images. This result relies on the robust camera pipeline (including black-level subtraction, demosaicing, white-balance) provided by libraw, and should be suitable for most projects.

If you need finer control over the exact radiance values, this behaviour can be overriden to merge RAW bayer images by setting the flag `demosaic_first=False`. This mode is useful when the camera is custom-calibrated and you have an exact correspondance between camera pixels with the scene luminance and/or color. Moreover, saturated pixels can be precisely identified before demosaicing. In this mode, a basic camera pipeline is reproduced with the following steps:

Subtract black level -> Merge image stack -> Color transformation -> White-balance

Demosaicing algorithms that are currently supported can be found at [this page](https://colour-demosaicing.readthedocs.io/en/latest/colour_demosaicing.bayer.html). Change the algorithm using `HDRutils.merge(..., demosaic_first=False, demosaic=*algorithm*)`


### Merge RAW bayer frames from non-RAW formats
If your camera provides RAW frames in a non-standard format, you can still merge them in the camera color-space without libraw processing

```python
files = ['file1.png', 'file2.png', 'file3.png']     # PNG bayer input files
HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw')[0]
HDRutils.imwrite('merged.exr', HDR_img)
```

### Alignment
While merging, some ghosting artifacts can be removed by setting `HDRutils.merge(..., align=True)`. This attempts homography alignment and corrects camera motion for still scenes. Unfortunately non-rigid motion requiring dense optical flow is not yet implemented.


### Exposure estimation
Exposure metadata from EXIF may be inaccurate and it may be benificial to estimate relative exposures directly from the image stack. Please see [our paper](https://www.cl.cam.ac.uk/research/rainbow/projects/exposure-estimation/) for details.

This feauture is currently disabled, and EXIF values are used by default. To enable exposure estimation, run `HDRutils.merge(..., estimate_exp='mst')`.

## Noise simulation
Generating realistic camera noise using calibrated parameters of real-world cameras is described [here](HDRutils/noise_modeling/).

## Citation
If you find this package useful, we would be grateful if you cite

    @inproceedings{hanji2020noise,
        author    = {Hanji, Param and Zhong, Fangcheng and Mantiuk, Rafa{\l} K.},
        title     = {Noise-Aware Merging of High Dynamic Range Image Stacks without Camera Calibration},
        booktitle = {Advances in Image Manipulation (ECCV workshop)},
        year      = {2020},
        publisher = {Springer},
        pages     = {376--391},
        url       = {http://www.cl.cam.ac.uk/research/rainbow/projects/noise-aware-merging/},
    }

    @ARTICLE{hanji2023exposures,
        author    = {Hanji, Param and and Mantiuk, Rafa{\l} K.},
        journal   = {IEEE Transactions on Computational Imaging}, 
        title     = {Efficient Approximation of Jacobian Matrices Involving a Non-Uniform Fast Fourier Transform (NUFFT)}, 
        year      = {2023},
        volume    = {},
        number    = {},
        pages     = {},
        doi       = {},
        url       = {https://www.cl.cam.ac.uk/research/rainbow/projects/exposure-estimation/},
    }
