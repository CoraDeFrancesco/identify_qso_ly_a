#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:56:39 2022

@author: user1
"""
# Imports --------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import peak_widths

# Setup ----------------------------------------------------------------------

plot_int = True

bnorm = -0.75
bmu = 1100
bsigma = 5
bshift = 1

bnorm2 = -0.4
bmu2 = 1102
bsigma2 = 5
bshift2 = 1


lnorm = -0.65
lmu = 1104
lsigma = 0.4
lshift = 1

lnorm2 = -0.65
lmu2 = 1090
lsigma2 = 0.4
lshift2 = 1

x = np.linspace(1000, 1200, 1000)
noise1 = np.random.normal(1, 0.02, size=len(x))
noise2 = np.random.normal(1, 0.07, size=len(x))

delta = 5 # Number of wavelength bins to fit in one gaussian abs line
perc_cut=0.8 # Between 0 and 1. Percentage of normalized flux abs must exceed.
wave_min=1060 # Min wavelength for finding peaks.
wave_max=1200 # Maximum wavelength for finding peaks
width=2 # Number of wavelength bins for a minimum to be considered a peak.
wmax=80 #Maximum width of abs lines in wavelength bins. The default is 5.
distance=5

# Functions ------------------------------------------------------------------

def gauss(wave, *p):
    
    norm, mu, sigma, shift = p
    
    curve = shift+norm*np.exp(-(wave-mu)**2/(2.*sigma**2))
    
    return(curve)

def get_line_mins(wave, norm_flux, wave_min=900, wave_max=1216, prom=0.05,\
                  distance=10, width=2, wmax=6, perc_cut=1.0, plot=True):
    """
    Get miminum abs line values in Ly-a region.

    Parameters
    ----------
    wave : arr
        Wavelength values.
    norm_flux : arr
        Normalized flux values.
    wave_min : float, optional
        Minimum wavelength for finding lines. The default is 900.
    wave_max : float, optional
        Maximum wavelength for finding lines. The default is 1216.
    prom : float, optional
        Minimum required prominence for lines. The default is 0.05.
    distance : float, optional
        Miminum distance in wavelength bins between potential abs lines. The default is 2.
    width : float, optional
        Minimun width of abs lines in wavelength bins. The default is 2.
    wmax : float, optional
        Maximum width of abs lines in wavelength bins. The default is 5.
    perc_cut : float, optional
        Between 0 and 1. Percentage of normalized flux abs must exceed. The default is 1.0.
    plot : bool, optional
        If true, plot the Ly-a region and mark the found line mins. The default is True.

    Returns
    -------
    Wavelength of line mins, fluxes of line mins.
    """
    wave_mask = np.where((wave >= wave_min) & (wave <= wave_max))
    
    #x = -1*norm_flux[wave_mask]
    x = -1*norm_flux
    
    border = -perc_cut*np.ones(x.size)
    peaks, properties = find_peaks(x, height=(border, 0), \
                                   distance=distance, width=(width, wmax), prominence=(prom))
    
    line_mask = np.where((wave[peaks] >= wave_min) & (wave[peaks] <= wave_max))
    
    line_mins_wave = wave[peaks][line_mask]
    line_mins_flux = -1*x[peaks][line_mask]
    line_mins_idxs = peaks[line_mask]
    
    if plot:
        
        
        plt.plot(wave, norm_flux, lw=1, ds='steps-mid')
    
        plt.plot(wave, -border, "--", color="gray")
        #plt.plot(border, ":", color="gray")
        plt.plot(wave[peaks], -x[peaks], "x")
        plt.xlim(wave_min, wave_max)
        plt.ylim(-0.1, max(norm_flux[wave_mask]))
        
        # plt.hlines(y=-1*properties["width_heights"], xmin=properties["left_ips"]+idx_min,
        #             xmax=properties["right_ips"]+idx_min, color = "C1")
        
        
        plt.show()
        plt.clf()
        
    return(line_mins_wave, line_mins_flux, line_mins_idxs)

def get_noise(data_wave, data_flux, wave_min, wave_max):
    
    reg_mask = np.where((data_wave < wave_max) & (data_wave > wave_min))
    reg_flux = data_flux[reg_mask]
    noise=np.std(reg_flux)
    
    return(noise)

# Simulated Observations -----------------------------------------------------

bline = gauss(x, bnorm, bmu, bsigma, bshift)
bline2 = gauss(x, bnorm2, bmu2, bsigma2, bshift2)
lline = gauss(x, lnorm, lmu, lsigma, lshift)
lline2 = gauss(x, lnorm2, lmu2, lsigma2, lshift2)

ly_min_mask = np.where(lline == min(lline))

conv_1 = bline*lline*lline2*noise1
conv_2 = bline2*lline*lline2*noise2

if plot_int:

    plt.figure(dpi=200)
    
    plt.plot(x, bline, color='red', alpha=0.5)
    plt.plot(x, bline2, color='purple', alpha=0.5)
    
    plt.plot(x, lline*lline2, label='Ly-a', lw=1, color='green')
    plt.plot(x,conv_1, label='Observed 1', color='red', lw=2)
    plt.plot(x,conv_2, label='Observed 2', color='purple', lw=2)
    
    plt.legend()
    
    plt.ylim(-0.01, 1.25)
    plt.xlim(1050, 1150)
    
    plt.show()
    plt.clf()

# print('Observed 1 broad min:', -bnorm)
# print('            Ly-a min:', conv_1[ly_min_mask])   
# print('Observed 2 broad min:', -bnorm2)
# print('            Ly-a min:', conv_2[ly_min_mask])   

# print('Ratio 1:', (-bnorm / conv_1[ly_min_mask]))
# print('Ratio 2:', (-bnorm2 / conv_2[ly_min_mask]))

# print('Ly ratio = ', ((-bnorm / conv_1[ly_min_mask])/(-bnorm / conv_2[ly_min_mask])))

# Find Noise -----------------------------------------------------------------

prom1 = 4*get_noise(x, conv_1, 1160, 1180)
prom2 = 4*get_noise(x, conv_2, 1160, 1180)

# Find Peaks -----------------------------------------------------------------

line_waves1, line_fluxes1, line_idxs1 = get_line_mins(x, conv_1, \
                                        perc_cut=perc_cut, wave_min=wave_min, \
                                        wave_max=wave_max, width=width, wmax=wmax,\
                                        prom=prom1, plot=True, distance=distance)
    
line_waves2, line_fluxes2, line_idxs2 = get_line_mins(x, conv_2, \
                                        perc_cut=perc_cut, wave_min=wave_min, \
                                        wave_max=wave_max, width=width, wmax=wmax,\
                                        prom=prom2, plot=True, distance=distance)
    
line_waves = [line_waves1, line_waves2]
line_fluxes = [line_fluxes1, line_fluxes2]
line_idxs = [line_idxs1, line_idxs2]
line_idxs = np.asarray(line_idxs)
flat_idxs = np.ndarray.flatten(line_idxs)

# Check to see if there is a matching Ly-a line for each peak

line_idx_matched = []

for i, idx_list in enumerate(line_idxs):
    
    for idx in idx_list:
        
        # check that we are within dist for all specs
        idx_min = idx - distance
        idx_max = idx + distance
        
        print('Checking for matches in range(', idx_min, idx_max, ')')
        print('     Options:', flat_idxs)
        # go to subsequent index lists
        matches = 0
        for fidx in flat_idxs:
            
            if fidx in range(idx_min, idx_max):
                print('     Found:', fidx)
                matches += 1
        print('     Total matches:', matches)
        if matches == len(line_idxs):
            line_idx_matched.append(idx)
            print('     Adding idx to matched list.')
        else:
            print('     Not enough matches. Disregarding idx.')
            
line_idx_matched = np.unique(np.asarray(line_idx_matched)) # need to only select one idx per line
    
    

# Find Broad Abs -------------------------------------------------------------

# line_widths1 = peak_widths(-1*conv_1, line_idxs1, rel_height=0.5)
# line_widths2 = peak_widths(-1*conv_2, line_idxs2, rel_height=0.5)

# plt.figure(dpi=200)
# plt.plot(conv_1, alpha=0.5)
# plt.plot(conv_2, alpha=0.5)
# plt.plot(line_idxs1, conv_1[line_idxs1], 'x')
# plt.plot(line_idxs2, conv_2[line_idxs2], 'x')
# plt.hlines(-line_widths1[1], line_widths1[2], line_widths1[3], color='red')
# plt.hlines(-line_widths2[1], line_widths2[2], line_widths2[3], color='purple')
# plt.show()
# plt.clf()
