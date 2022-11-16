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
from scipy.interpolate import UnivariateSpline

from specutils import Spectrum1D
from specutils.fitting import find_lines_derivative
from specutils.fitting import find_lines_threshold
from astropy import units as u
from specutils import SpectralRegion
from specutils.fitting import estimate_line_parameters
from specutils.manipulation import extract_region
from astropy.modeling import models
from specutils.manipulation import noise_region_uncertainty
from specutils.manipulation import median_smooth

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
lsigma2 = 0.45
lshift2 = 1

lnorm3 = -0.45
lmu3 = 1107
lsigma3 = 0.45
lshift3 = 1

x = np.linspace(1000, 1200, 700)
noise1 = np.random.normal(1, 0.04, size=len(x))
noise2 = np.random.normal(1, 0.07, size=len(x))

delta = 12 # Number of wavelength bins to fit in one gaussian abs line
perc_cut=0.8 # Between 0 and 1. Percentage of normalized flux abs must exceed.
wave_min=1060 # Min wavelength for finding peaks.
wave_max=1200 # Maximum wavelength for finding peaks
width=2 # Number of wavelength bins for a minimum to be considered a peak.
wmax=50 #Maximum width of abs lines in wavelength bins. The default is 5.
distance=2 #Miminum distance in wavelength bins between potential abs lines.
n_prom = 2 #how many times the noise level a peak must be

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
    Wavelength of line mins, fluxes of line mins indexs of line mins.
    """
    wave_mask = np.where((wave >= wave_min) & (wave <= wave_max))
    
    #x = -1*norm_flux[wave_mask]
    x = -1*norm_flux
    
    border = -perc_cut*np.ones(x.size)
    peaks, properties = find_peaks(x, height=(border, 0), \
                                   width=(width, wmax), prominence=(prom))
    
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

#%% Building specutils get line mins ------------------------------------------

def get_line_mins_specutils(wave, flux, delta):
    
    wave = u.Quantity(wave, u.Angstrom)
    flux = flux*u.dimensionless_unscaled
    flux_fit = flux-1
    spectrum = Spectrum1D(flux=flux_fit, spectral_axis=wave)
    

    noise_reg = SpectralRegion(1025*u.angstrom, 1030*u.Angstrom)
    spectrum = noise_region_uncertainty(spectrum, noise_reg)
    spectrum_smooth = median_smooth(spectrum, width=5)
    
    plt.figure(dpi=200)
    plt.plot(spectrum_smooth.spectral_axis, spectrum_smooth.flux)
    plt.title('smoothed')
    plt.show()
    plt.clf()
    
    table_out = find_lines_derivative(spectrum_smooth)
    #print(table_out)
    
    line_centers = table_out['line_center']
    line_type = table_out['line_type']
    line_center_index = np.asarray(table_out['line_center_index'])
    
    abs_idxs = np.where(line_type == 'absorption')
    
    line_mins_wave = line_centers[abs_idxs]
    line_mins_idxs = line_center_index[abs_idxs]
    line_mins_flux = flux[line_mins_idxs].value
    
    
    #for each line, get the std of its gaussian profile
    #ignore the super tiny and super big ones
    
    keep_idxs = []
    line_params_save = []
    sigs_list=[]
    for i, line_wave in enumerate(line_mins_wave):
        
        
        
        min_idx = delta
        max_idx = len(wave) - 1 - delta
        if line_mins_idxs[i] <= min_idx:
            wave_min = wave[0]
            wave_max = wave[line_mins_idxs[i] + delta]
        elif line_mins_idxs[i] >=  max_idx:
            wave_min = wave[line_mins_idxs[i] - delta]
            wave_max = wave[-1]
        else:   
            wave_min = wave[line_mins_idxs[i] - delta]
            wave_max = wave[line_mins_idxs[i] + delta]
        
        sub_region = SpectralRegion(wave_min, wave_max)
        sub_spec = extract_region(spectrum, sub_region)
        
        line_params = estimate_line_parameters(sub_spec, models.Gaussian1D())
        # amplitude = line_params.amplitude
        #print('amplitude = ', amplitude.value)
        
        
        #print(line_mins_flux[i])
        if (line_mins_flux[i] < 0.8):
            # and (line_params.stddev.value > 1)\
            # and (line_params.stddev.value < 4):
                 # and (line_params.amplitude < 0.):
                     
            plt.figure(dpi=100)
            plt.plot(sub_spec.spectral_axis, sub_spec.flux)
            plt.title('sub spec')
            plt.show()
            plt.clf()
            
            keep_idxs.append(i)
            line_params_save.append(line_params)
            # sigs_list.append(line_params.stddev.value)
            #print(line_params)
    # plt.figure(dpi=200) 
    # plt.hist(sigs_list)
    # plt.show()
    # plt.clf()       
                
    line_mins_wave = line_centers[abs_idxs][keep_idxs].value
    line_mins_idxs = line_center_index[abs_idxs][keep_idxs]
    line_mins_flux = line_mins_flux[keep_idxs]

    return(line_mins_wave, line_mins_flux, line_mins_idxs, line_params_save)


    
    

#%% Simulated Observations -----------------------------------------------------

bline = gauss(x, bnorm, bmu, bsigma, bshift)
bline2 = gauss(x, bnorm2, bmu2, bsigma2, bshift2)
lline = gauss(x, lnorm, lmu, lsigma, lshift)
lline2 = gauss(x, lnorm2, lmu2, lsigma2, lshift2)
lline3 = gauss(x, lnorm3, lmu3, lsigma3, lshift3)

ly_min_mask = np.where(lline == min(lline))

conv_1 = bline*lline*lline2*noise1*lline3
conv_2 = bline2*lline*lline2*noise2*lline3

if plot_int:

    plt.figure(dpi=200)
    
    plt.plot(x, bline, color='red', alpha=0.5)
    plt.plot(x, bline2, color='purple', alpha=0.5)
    
    #plt.plot(x, lline*lline2, label='Ly-a', lw=1, color='green')
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

prom1 = n_prom*get_noise(x, conv_1, 1160, 1180)
prom2 = n_prom*get_noise(x, conv_2, 1160, 1180)

# Find Peaks -----------------------------------------------------------------



#%% Testing specutils get line mins

# line_waves1, line_fluxes1, line_idxs1, lps1 = get_line_mins_specutils(x, conv_1, delta)

# line_waves2, line_fluxes2, line_idxs2, lps2 = get_line_mins_specutils(x, conv_2, delta)





#%%


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
#flat_idxs = np.hstack(line_idxs)

# Check to see if there is a matching Ly-a line for each peak

def match_lines(line_idxs, distance):
    """
    Match Ly-a abs lines from two specs.

    Parameters
    ----------
    line_idxs : arr
        Array of line indexes of the form [idx_arr1, idx_arr2, ..., idx_arrN].
    distance : float
        Miminum distance in wavelength bins between potential abs lines.

    Returns
    -------
    Array of matched Ly-a indexes.

    """
    
    flat_idxs = np.hstack(line_idxs)
    
    line_idx_matched = []
    
    for i, idx_list in enumerate(line_idxs):
        matched_list_i = []
    
        for idx in idx_list:
            
            # check that we are within dist for all specs
            idx_min = idx - distance
            idx_max = idx + distance
            
            # print('Checking for matches in range(', idx_min, idx_max, ')')
            # print('     Options:', flat_idxs)
            # go to subsequent index lists
            matches = 0
            for fidx in flat_idxs:
                
                if fidx in range(idx_min, idx_max):
                    # print('     Found:', fidx)
                    matches += 1
            # print('     Total matches:', matches)
            if matches == len(line_idxs):
                matched_list_i.append(idx)
                # print('     Adding idx to matched list.')
            # else:
                # print('     Not enough matches. Disregarding idx.')
        line_idx_matched.append(matched_list_i)
    
    return(line_idx_matched)

  
line_idx_matched = match_lines(line_idxs, distance)  
    

# Plot Matches ---------------------------------------------------------------

plt.figure(dpi=200)
plt.plot(x, conv_1, alpha=0.5)
plt.plot(x, conv_2, alpha=0.5)
plt.plot(x[line_idxs1], conv_1[line_idxs1], 'x', label='')
plt.plot(x[line_idxs2], conv_2[line_idxs2], 'x')
# plt.hlines(-line_widths1[1], line_widths1[2], line_widths1[3], color='red')
# plt.hlines(-line_widths2[1], line_widths2[2], line_widths2[3], color='purple')
plt.plot(x[line_idx_matched[0]], np.ones(len(line_idx_matched[0])), '*')
plt.show()
plt.clf()

#%% Smooth specs, fit spline ---------------------------------------------------

#Mask out positions of the Ly-a lines

lya_masks = []

for line_list in line_idx_matched:
    lya_mask = np.asarray([])
    for idx in line_list:
        min_val = idx - delta
        max_val = idx + delta
        idxs = np.linspace(min_val, max_val, (2*delta)+1)
        lya_mask = np.concatenate((lya_mask, idxs))
    lya_mask = np.unique(lya_mask)
    lya_masks.append(lya_mask)

spline_mask_1 = []
spline_mask_2 = []
for i in range(0, len(x)):
    if i not in lya_masks[0]:
        spline_mask_1.append(i)
    if i not in lya_masks[1]:
        spline_mask_2.append(i)

        

#Spline fitting with smoothing s and degree k
spl1 = UnivariateSpline(x[spline_mask_1], conv_1[spline_mask_1], s=1, k=3)
spl2 = UnivariateSpline(x[spline_mask_2], conv_2[spline_mask_2], s=1, k=3)

plt.figure(dpi=200)
# plt.plot(x[spline_mask_1], conv_1[spline_mask_1], alpha=0.5, marker='o', ls='', ms=2)
# plt.plot(x[spline_mask_2], conv_2[spline_mask_2], alpha=0.5, marker='o', ls='', ms=2)
plt.plot(x, conv_1, alpha=0.5)
plt.plot(x, conv_2, alpha=0.5)
plt.plot(x, spl1(x))
plt.plot(x, spl2(x))

plt.show()
plt.clf()
