# HDRutils

Some utility functions to generate HDR images from a sequence of exposure time or gain modulated images. You can find a separate Readme describing some functinos for noise simulations [here](HDRutils/noise_modeling).

## Installation
To download HDRUtils, use Pypi via pip:

    pip install HDRutils

If you prefer cloning this repository, install the dependencies using pip:
    
    pip clone https://github.com/gfxdisp/HDRutils.git
    cd HDRutils
    pip install -e .

### Additional dependency
You will need the [FreeImage plugin](https://imageio.readthedocs.io/en/stable/format_exr-fi.html) for reading and writing OpenEXR images:

    imageio_download_bin freeimage

## Usage
Just import HDRutils and call the required functions

### Reading and writing
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

### Merge input images
The [rawpy](https://github.com/letmaik/rawpy) wrapper is used to read RAW images. [Noise-aware merging](https://www.cl.cam.ac.uk/research/rainbow/projects/noise_aware_merging/) is performed using the Poisson-noise optimal estimator. The generated HDR image is linearly related to the scene radiance

```python
files = ['file1.dng', 'file2.dng', 'file3.dng']		# RAW input files
HDR_img = HDRutils.merge(files)
HDRutils.imwrite('merged.exr', HDR_img)
```

The default function processes each image individually using [libraw](https://www.libraw.org/) and then merges the RGB images. This behaviour can be overriden to merge RAW bayer image by setting the flag `demosaic_first=False`.

### Merge RAW bayer frames from non-RAW formats
If your camera provides RAW frames in a non-standard format, you can still merge them in the camera color-space without libraw processing

```python
files = ['file1.png', 'file2.png', 'file3.png']     # PNG bayer input files
HDR_img = HDRutils.merge(files, demosaic_first=False, color_space='raw')
HDRutils.imwrite('merged.exr', HDR_img)
```

### Alignment
While merging, some ghosting artifacts an be removed by setting `HDRutils.merge(..., align=True)`. This attempts homography alignment and corrects camera motion for still scenes.

### Estimate exposure times
The exposure times reported in EXIF camera metadata may be inaccurate. The merge function solves a linear least squares problem to estimate most suitable exposure times. This setting can be disabled by `HDRutils.merge(..., estimate_exp=False)`. The default behaviour is to use the brightest 10% of the pixels for estimation.

### Exposure estimation
Exposure metadata from EXIF may be inaccurate. The default behaviour is to estimate relative exposures directly from the image stack. If you are confident that metadata is correct, disable exposure estimation by specifying `HDRutils.merge(..., estimate_exp=None)`.

For robustness, the estimation includes an iterative outlier removal procedure which may take a couple of minutes to converge especially for large images and deep stacks. You can override this by `HDRutils.merge(..., outlier=None)`. For best results, supply the exact camera (instance of `HDRutils.NormalNoise`). Otherwise a default camera that works reasonably well for tested images will be used.

## Citation
If you find this package useful, please cite

    @inproceedings{hanji2020noise,
        author    = {Hanji, Param and Zhong, Fangcheng and Mantiuk, Rafa{\l} K.},
        title     = {Noise-Aware Merging of High Dynamic Range Image Stacks without Camera Calibration},
        booktitle = {Advances in Image Manipulation (ECCV workshop)},
        year      = {2020},
        publisher = {Springer},
        pages     = {376--391},
        url       = {http://www.cl.cam.ac.uk/research/rainbow/projects/noise-aware-merging/},
    }
