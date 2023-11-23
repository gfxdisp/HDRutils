#!/usr/bin/env python
import os.path as op
import numpy as np
from scipy.optimize import curve_fit
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
#import pyexr
import image_io

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

def deglare(I, gParams):
    I_deconv = np.zeros_like(I)
    rho_x, rho_y = create_rho_2D(I.shape[:2], 0.5, 0.5)
    rho_x, rho_y = fftshift(rho_x), fftshift(rho_y)
    rho = np.sqrt(rho_x**2 + rho_y**2)
    mtf_filter = gauss2(rho, *gParams)
    for cc in range(1):
        If = fftshift(fft2(I[:, :, cc]))
        I_deconv[:, :, cc] = np.abs(ifft2(ifftshift(If / mtf_filter)))
    return I_deconv

def createFit(mtf_freq, mtf_amp):
    #p0 = [1.0, 0.0, 0.0814, 0.6275, 0.138, 0.1001]
    p0 = [0.5802, -0.01012, 0.1046, 0.4896, 0.3853, 0.9892]
    low_bounds = [-np.inf, -np.inf, 0.0, -np.inf, -np.inf, 0.0]
    up_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    params, _ = curve_fit(gauss2, xdata=mtf_freq, ydata=mtf_amp, p0=p0, bounds=(low_bounds, up_bounds), method='trf')
    return params

def plotFreqVsAmp(mtf_freq, mtf_amp, lowest_amp, gParams):
        plt.figure('Deglaring GM Fit')
        plt.scatter(mtf_freq, mtf_amp, label="mtf_amp vs. mtf_freq", c='k', marker='.')
        plt.hlines(y=lowest_amp, xmin=0.0, xmax=0.5, linewidth=2, color='k', linestyle='--', alpha=0.5)
        plt.plot(mtf_freq, gauss2(mtf_freq, *gParams), label="GM Fit", c='b', linewidth=2, alpha=0.5)
        plt.ylim([0.0, 1.0])
        plt.xlabel('mtf_freq')
        plt.ylabel('mtf_amp')
        plt.legend(loc='upper right')
        plt.grid(True)

# deglare an image with a single channel (using the luminance in the SFR)
def deglare_channel(img, sfr, lowest_amp=0.5, freq_factor=1.0, amp_factor=1.0, plot=False):
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    # Read data
    T = np.genfromtxt(sfr, delimiter=',')
    mtf_freq, mtf_amp = T[:, 0], T[:, 4]

    # Multiplicative factors
    mtf_freq *= freq_factor
    mtf_amp  *= amp_factor

    # Filter data
    mtf_amp = mtf_amp[mtf_freq <= 0.5]
    last_valid_index = np.where(mtf_freq <= 0.35)[0][-1]
    mtf_amp[last_valid_index:] = mtf_amp[last_valid_index]
    mtf_freq = mtf_freq[mtf_freq <= 0.5]
    mtf_amp_clipped = np.clip(mtf_amp, lowest_amp, 1.0)
    # Fit
    gParams = createFit(mtf_freq, mtf_amp_clipped)
    print("Fitted GM Parameters:", gParams)
    if plot is True:
        plotFreqVsAmp(mtf_freq, mtf_amp, lowest_amp, gParams)
    img_deconv = deglare(img, gParams)
    return img_deconv

# deglare an image per channel (using the RGB curves of SFR instead of luminance)
def deglareRGB_img(img, sfr, lowest_amp=0.5, freq_factor=1.0, amp_factor=1.0, plot=False):
    # Read data
    T = np.genfromtxt(sfr, delimiter=',')
    gParams, img_deconv = [], []
    for irgb in [0, 1, 2]:
        mtf_freq, mtf_amp = T[:, 0], T[:, irgb+1]

        # Multiplicative factors
        mtf_freq *= freq_factor
        mtf_amp  *= amp_factor

        # Filter data
        mtf_amp = mtf_amp[mtf_freq <= 0.5]
        last_valid_index = np.where(mtf_freq <= 0.35)[0][-1]
        mtf_amp[last_valid_index:] = mtf_amp[last_valid_index]
        mtf_freq = mtf_freq[mtf_freq <= 0.5]
        mtf_amp_clipped = np.clip(mtf_amp, lowest_amp, 1.0)
        # Fit
        gParams.append(createFit(mtf_freq, mtf_amp_clipped))
        print("Fitted GM Parameters:", gParams[irgb])
        if plot is True:
            plotFreqVsAmp(mtf_freq, mtf_amp, lowest_amp, gParams[irgb])
        img_deconv.append(deglare(img[:,:,irgb:irgb+1], gParams[irgb]))
    img_deconv = np.moveaxis(np.squeeze(img_deconv), 0, -1)
    return img_deconv

