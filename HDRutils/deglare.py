#!/usr/bin/env python
import os.path as op
import numpy as np
from scipy.optimize import curve_fit
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import json

def create_rho_2D(im_size, nf_rows, nf_cols):
    """
    Create frequency coeffs.
    Useful for constructing Fourier-domain filters based on OTF or CSF data.
    """
    half_size = np.floor(np.array(im_size) / 2).astype(int)
    odd = np.mod(im_size, 2)
    freq_step = [nf_rows, nf_cols] / half_size

    if odd[1]:
        xx = np.concatenate([
            np.linspace(0, nf_cols, half_size[1] + 1),
            np.linspace(-nf_cols, -freq_step[1], half_size[1])
        ])
    else:
        xx = np.concatenate([
            np.linspace(0, nf_cols - freq_step[1], half_size[1]),
            np.linspace(-nf_cols, -freq_step[1], half_size[1])
        ])

    if odd[0]:
        yy = np.concatenate([
            np.linspace(0, nf_rows, half_size[0] + 1),
            np.linspace(-nf_rows, -freq_step[0], half_size[0])
        ])
    else:
        yy = np.concatenate([
            np.linspace(0, nf_rows - freq_step[0], half_size[0]),
            np.linspace(-nf_rows, -freq_step[0], half_size[0])
        ])
    rho_x, rho_y = np.meshgrid(xx, yy)
    return rho_x, rho_y

def gauss2(rho, a1, b1, c1, a2, b2, c2):
    term1 = a1 * np.exp(-((rho - b1) / c1) ** 2)
    term2 = a2 * np.exp(-((rho - b2) / c2) ** 2)
    return term1 + term2

def deglare(I, gParams, freq_factor=2.0):
    I_deconv = np.zeros_like(I)
    rho_x, rho_y = create_rho_2D(I.shape[:2], 0.5, 0.5)
    rho_x, rho_y = fftshift(rho_x), fftshift(rho_y)
    rho = np.sqrt(rho_x**2 + rho_y**2) / freq_factor
    mtf_filter = gauss2(rho, *gParams)
    for cc in range(1):
        If = fftshift(fft2(I[:, :, cc]))
        I_deconv[:, :, cc] = np.abs(ifft2(ifftshift(If / mtf_filter)))
    return I_deconv

def bayer2rggb(bayer): # input: 2D array with bayer pattern [[r,g],[g,b]]
	r = bayer[0::2, 0::2]  # red pixels
	g1 = bayer[0::2, 1::2]  # green pixels 1
	g2 = bayer[1::2, 0::2]  # green pixels 2
	b = bayer[1::2, 1::2]  # blue pixels
	return np.array([r, g1, g2, b]) # output: array with 4x 2D arrays (rggb)

def rggb2bayer(rggb): # input: array with 4x 2D arrays (rggb)
	bayer = np.zeros([rggb.shape[1]*2, rggb.shape[2]*2])
	bayer[0::2, 0::2] = rggb[0]
	bayer[0::2, 1::2] = rggb[1]
	bayer[1::2, 0::2] = rggb[2]
	bayer[1::2, 1::2] = rggb[3]
	return bayer # ouput: 2D array with bayer pattern [[r,g],[g,b]]

# deglare an image with a single channel (using the luminance in the MTF).
def deglare_channel(img, gParams):
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
    img_deconv = deglare(img, gParams)
    return img_deconv[:,:,0]

# deglare an image with a bayer pattern RGGB
def deglare_bayer(bayer_img, mtf_json):
    # Read GMM Fit params (Luminance)
    with open(mtf_json) as f:
        gParams_dict = json.load(f)
    gParams = gParams_dict["Y"]

    rggb = bayer2rggb(bayer_img)
    for irggb in range(4):
        rggb[irggb] = deglare_channel(rggb[irggb], gParams)
    bayer_ret = rggb2bayer(rggb)
    return bayer_ret
