# Noise simulatations
We demonstrate how to use a simple noise model to generate sythetic captures by providing a ground truth HDR image, along with exposure time (in seconds) and gain (ISO 100 corresponds to gain 1). For best results, the input HDR image should not be normalized or transformed to a different (non-native) colorspace. The result of a call to the `simulate()` function is a noisy image without any quantization.

## Darktable cameras
Simulate using a parameters from a large open-source database linked with the Darktable database. The detailed noise model can be found [here](https://www.darktable.org/2012/12/profiling-sensor-and-photon-noise/).

```python
hdr_file = 'example.exr'
img = HDRutils.imread(hdr_file)
camera_make, camera_model = 'Canon', 'EOS-1Ds'
exp_time, iso = 1, 100

model = HDRutils.NormalNoise()
noisy_img = model.simulate(img, camera_make, camera_model, exp_time, iso)
```

## Cameras calibrated in Cambridge
Simulate Cambridge cameras using parameters obtained from calibration with the noise target. Ideally, this simulation should be performed on RAW Bayer data before deomsaicing since calibration was performed on Bayer data. Currently the following cameras are supported:
- SonyA7r1
- SonyA7r3
- CanonT1
- SamsungS9

```python
hdr_file = 'example.exr'
img = HDRutils.imread(hdr_file)
preset = 'SonyA7r1'
exp_time, iso = 1, 100

model = HDRutils.PoissonNormalNoise(preset)
noisy_img = model.simulate(img, exp_time, iso)
```

## TODO: NoiseFlow
