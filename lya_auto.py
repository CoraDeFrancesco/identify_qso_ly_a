#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 19:05:51 2022

@author: CoraAnn

Automatically identify and filter out ly-a absorption in quasar spectra.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
#import os
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

#-----------------------------------------------------------------------------
# Setup
#-----------------------------------------------------------------------------

obj_name = 'J085825'
spec_dir = 'specs/J085825/' # data directory (with /)
spec_file_1 = 'spec-0468-51912-0036-dered.txt' # spectrum 1 file name
spec_file_2 = 'spec-3815-55537-0910.dr9' # spectrum 2 file name
spec_mjd_1 = '51912' # MJD of spectrum 1
spec_mjd_2 = '55537' # MjD of spectrum 2

z = 2.8684 # redshift of object (float)

delta = 5 # Number of wavelength bins to fit in one gaussian abs line
perc_cut=0.75 # Between 0 and 1. Percentage of normalized flux abs must exceed.
wave_min=1060 # Min wavelength for finding peaks.
wave_max=1216 # Maximum wavelength for finding peaks
width=2 # Number of wavelength bins for a minimum to be considered a peak.
wmax=80 #Maximum width of abs lines in wavelength bins. The default is 5.

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def align_data(data_wave, data_flux, data_err, data_names=None, wave_min=1000.,\
               wave_max=3000., get_res=True, save=False, save_dir=None, res=None,\
               interp_wave=0):
    """
    Parameters
    ----------
    data_wave : arr
        DESCRIPTION.
    data_flux : arr
        DESCRIPTION.
    data_err : arr
        DESCRIPTION.
    data_names : arr, optional
        Names of the spectra to use when saving. Generally of the form
        'spec-plate-mjd-fiber'
    wave_min : float, optional
        DESCRIPTION. The default is 1000.
    wave_max : float, optional
        DESCRIPTION. The default is 3000.
    get_res : bool, optional
        DESCRIPTION. The default is True, in which case the resolution will be 
        calculated for each spectrum, and the minimum will be used.
        If False, the default resolution of 1 wavelength bin per Angstrom
        will be used. The user can set a resolution with kwarg 'res'.
    save : bool, optional
        DESCRIPTION. The default is False.
    save_dir : str, optional
        DESCRIPTION. The default is None.
    res : float, optional
        Resolution to use. 'get_res' must be set to 'False'.
    interp_wave : arr, optional
        If provided, use this wavelength array for all spectra instead of
        calculating the resolution.

    Returns
    -------
    interp_wave : arr
        Aligned wavebins.
    interp_fluxes : arr
        Fluxes at aligned wavebins (dimension = numspecs).
    interp_errs : arr
        Errors at aligned wavebins (dimension = numspecs).

    """
    
    numspecs = len(data_wave)
    
    if get_res: #calculate the minimum resolution of the sample
        lengths = [] #number of data points in each spectrum
        coverage = [] #wavelength coverage of each spectrum
        resolutions = [] #resolution of each spectrum
        for i in data_wave: #calculate the wavelength coverage of each spec
            length=len(i)
            lengths.append(length)
            delta = max(i) - min(i)
            coverage.append(delta)
        for idx, cov in enumerate(coverage): #calculate the resolution
            res = lengths[idx]/cov
            resolutions.append(res)
        res=min(resolutions)
        print('Default resolution (1 wavebin per A) reset to', res)
    elif res:
        res = res
    else:
        res=1
    
    if interp_wave:
        length = len(interp_wave)
        delta = max(interp_wave) - min(interp_wave)
        res = length / delta
        print('Resolution of provided wave arr is :', np.round(res,2), 'wavebins/Ang.')
    else:
        num_pts = res * (wave_max - wave_min)
        num_pts = int(np.round(num_pts, 0))
        interp_wave = np.linspace(wave_min, wave_max, num_pts)
    
    interp_fluxes = []
    interp_errs = []
    
    for i in range(0, numspecs):        
    
        aligned_flux = np.interp(interp_wave, data_wave[i], data_flux[i], left=0, right=0)
        aligned_err = np.interp(interp_wave, data_wave[i], data_err[i], left=0, right=0)
        
        interp_fluxes.append(aligned_flux)
        interp_errs.append(aligned_err)
        
        if save:
            
            save_name = str(data_names[i]) + str('_aligned.dat')
    
            save_data = np.array((interp_wave, aligned_flux, aligned_err)).T
            header = 'interp_wave, aligned_flux, aligned_err'
    
            np.savetxt((save_dir + save_name), save_data, header=header, fmt='%s')
            
    interp_fluxes = np.asarray(interp_fluxes, dtype='float')
    interp_errs = np.asarray(interp_errs, dtype='float')
    
    return(interp_wave, interp_fluxes, interp_errs)

def get_avg_flux_val(wave, flux, center, width):
    """
    Get the average flux value centered at some wavelength.

    Parameters
    ----------
    wave : arr
        Wavlength array for spectrum.
    flux : arr
        Flux array for spectrum.
    center : float
        Wavelength for which you want to calculate the average flux.
        Must be in same units as wave array.
    width : float
        Delta wavelength units over which to compute the average.

    Returns
    -------
    Average flux value within range.

    """
    
    avg_flux_val = 1
    
    wave_min = center - width
    wave_max = center + width
    
    wave_mask = np.where((wave >= wave_min) & (wave <= wave_max))
    
    avg_flux_val = np.median(flux[wave_mask])
    
    return(avg_flux_val)

def norm_specs(waves, fluxes, norm_point=1300, width=1):
    """
    Normalize spectra based on flux value at a certain wavelength.

    Parameters
    ----------
    waves : arr
        Wavelength arrays (1 for each spec).
    fluxes : arr
        Flux arrays.
    norm_point : float, optional
        Wavelength at which to normalize the spectra. The default is 1300.
    width : float, optional
        Width (in wavelength units) over which to average the flux values
        around the norm point. The default is 1.

    Returns
    -------
    wavelength arrays (unchanged), normalized fluxes.

    """
    norm_fluxes = []
    
    for i, flux in enumerate(fluxes):
        
        norm = get_avg_flux_val(waves[i], flux, center=norm_point, width=width)
        norm_flux = flux / norm
        norm_fluxes.append(norm_flux)
        
    return(waves, norm_fluxes)

def get_line_mins(wave, norm_flux, wave_min=900, wave_max=1216, \
                  distance=2, width=2, wmax=6, perc_cut=1.0, plot=True):
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
                                   distance=distance, width=(width, wmax))
    
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

def gauss(wave, *p):
    
    norm, mu, sigma, shift = p
    
    curve = shift+norm*np.exp(-(wave-mu)**2/(2.*sigma**2))
    
    return(curve)

#-----------------------------------------------------------------------------
# Execution
#-----------------------------------------------------------------------------

# Load Data ------------------------------------------------------------------

spec_files = [spec_file_1, spec_file_2] # file names
spec_mjds = [spec_mjd_1, spec_mjd_2] # mjds
spec_labels =  [(obj_name + ' MJD: ' + spec_mjd_1), (obj_name + ' MJD: ' + spec_mjd_2)]

obj_specs = [] # data!

for spec in spec_files:
    data = np.loadtxt((spec_dir+spec)).T
    obj_specs.append(data)

# Breaking it up for fun! Will probably regret this later. :)    
spec0 = obj_specs[0]
spec1 = obj_specs[1]

# Overplot data --------------------------------------------------------------

plt.figure(dpi=200)

plt.plot(spec0[0]/(1 + z), spec0[1], lw=1, alpha=0.7, \
         label=spec_labels[0], ds='steps-mid', color='blue')
plt.plot(spec1[0]/(1 + z), spec1[1], lw=1, alpha=0.7, \
         label=spec_labels[1], ds='steps-mid', color='red')

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel(r'Flux ($10^{-17}$ erg/cm$^2$/s/A)')
plt.title('Data')

plt.xlim(wave_min, 2000)
plt.ylim(bottom=-0.1, top=25)

plt.legend(fontsize=8)
plt.savefig((spec_dir+obj_name+'_overplot_specs.png'), format='png')
plt.show()

plt.clf()

# Align in wavelength bins ---------------------------------------------------

data_waves = [spec0[0]/(1 + z), spec1[0]/(1 + z)]
data_fluxes = [spec0[1], spec1[1]]
data_errs = [spec0[2], spec1[2]]
data_names = spec_labels

align_wave, align_fluxes, align_errs =  align_data(data_waves, data_fluxes, data_errs, \
                       data_names=data_names, wave_min=800, \
                       wave_max=2300, get_res=True, save=False)

# Check alignment ------------------------------------------------------------

plt.figure(dpi=200)

plt.plot(align_wave, align_fluxes[0], lw=1, alpha=0.7, \
         label=spec_labels[0], ds='steps-mid', color='blue')
plt.plot(align_wave, align_fluxes[1], lw=1, alpha=0.7, \
         label=spec_labels[1], ds='steps-mid', color='red')

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel(r'Flux ($10^{-17}$ erg/cm$^2$/s/A)')
plt.title('Aligned Specs')

plt.xlim(wave_min, wave_max)
plt.ylim(bottom=-0.1, top=25)

plt.legend(fontsize=8)

plt.show()

plt.clf()


# Crude normalization --------------------------------------------------------

norm_wave, norm_flux = norm_specs((align_wave, align_wave), align_fluxes)

# Check normalization --------------------------------------------------------

plt.figure(dpi=200)

plt.axhline(1, color='black', alpha=0.8)
plt.plot(norm_wave[0], norm_flux[0], lw=1, alpha=0.7, \
         label=spec_labels[0], ds='steps-mid', color='blue')
plt.plot(norm_wave[1], norm_flux[1], lw=1, alpha=0.7, \
         label=spec_labels[1], ds='steps-mid', color='red')

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel('Normalized Flux')
plt.title('Normalized Specs')

plt.xlim(wave_min, wave_max)
plt.ylim(bottom=-0.1, top=2)

plt.legend(fontsize=8)

plt.show()

plt.clf()

# Find Peaks -----------------------------------------------------------------

line_waves1, line_fluxes1, line_idxs1 = get_line_mins(norm_wave[0], norm_flux[0], \
                                        perc_cut=perc_cut, wave_min=wave_min, \
                                        wave_max=wave_max, width=width, wmax=wmax,\
                                        plot=True)
    
line_waves2, line_fluxes2, line_idxs2 = get_line_mins(norm_wave[1], norm_flux[1], \
                                        perc_cut=perc_cut, wave_min=wave_min, \
                                        wave_max=wave_max, width=width, wmax=wmax,\
                                        plot=True)
    
line_waves = [line_waves1, line_waves2]
line_fluxes = [line_fluxes1, line_fluxes2]
line_idxs = [line_idxs1, line_idxs2]

# Fit Ly-a Lines -------------------------------------------------------------

# fit parameters for both observations

fit_waves = []
fit_fluxes = []
err_waves = []
err_fluxes = []

for spec_idx in [0,1]:
    
    # fit parameters for each observation
    
    fit_waves_spec = []
    fit_fluxes_spec = []
    err_waves_spec = []
    err_fluxes_spec = []
    
    for i, line_idx in enumerate(line_idxs[spec_idx]):
        
        line_wave = line_waves[spec_idx][i]
        
        xdata = norm_wave[spec_idx][line_idx-delta:line_idx+delta]
        ydata = norm_flux[spec_idx][line_idx-delta:line_idx+delta]
        
        p0 = [-1, line_wave, 0.5, 1] # initial guess (norm, mu, sigma, shift)
        try:
            popt, pcov = curve_fit(gauss, xdata, ydata, p0=p0)
            fit_flux = gauss(xdata, *popt)
        
            fit_waves_spec.append(xdata)
            fit_fluxes_spec.append(fit_flux)
        except RuntimeError:
            print('RuntimeError for line', line_wave)
            err_waves_spec.append(line_wave)
            err_fluxes_spec.append(norm_flux[spec_idx][line_idx])
            
    err_waves_spec = np.asarray(err_waves_spec) 
    
    fit_waves.append(fit_waves_spec)
    fit_fluxes.append(fit_fluxes_spec)
    err_waves.append(err_waves_spec)
    err_fluxes.append(err_fluxes_spec)

# Plot Ly-a fits -------------------------------------------------------------
    
    
plt.figure(dpi=200)

plt.plot(norm_wave[0], norm_flux[0], alpha=0.4, color='blue', ds='steps-mid')
plt.plot(norm_wave[1], norm_flux[1], alpha=0.4, color='red', ds='steps-mid')
plt.plot(line_waves1, line_fluxes1, 'b*')
plt.plot(line_waves2, line_fluxes2, 'r*')
plt.axhline(1, color='black', alpha=0.5, ls='--', label='Norm')
plt.axhline(perc_cut, color='red', alpha=0.5, ls='--', label='Percent Cutoff')
plt.plot(err_waves[0], err_fluxes[0], 'gx', label='Fit Error')
plt.plot(err_waves[1], err_fluxes[1], 'gx')

for spec_idx in [0,1]:
    
    colors = ['blue', 'red']
    
    for i, flux in enumerate(fit_fluxes[spec_idx]):
        
        plt.plot(fit_waves[spec_idx][i], flux, 'r-', alpha=0.7, color=colors[spec_idx], lw=2)

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel(r'Normalized Flux')
plt.title('Fit Ly-a Lines')
    
plt.legend(fontsize=8)

plt.xlim(wave_min, wave_max)
# plt.xlim(1080, 1140)
plt.ylim(-0.1, 2)

plt.savefig((spec_dir+obj_name+'_fit_lines.png'), format='png')

plt.show()
plt.clf()

# Remove Lines ---------------------------------------------------------------

for spec_idx in [0,1]:
    
    for i, flux in enumerate(fit_fluxes[spec_idx]):
        
        line_mask = []
        
        for j, wavebin in enumerate(fit_waves[spec_idx][i]):
        
            wavebin_mask = np.where(norm_wave[spec_idx] == wavebin)[0][0]
            line_mask.append(wavebin_mask)
        
        norm_flux[spec_idx][line_mask] = norm_flux[spec_idx][line_mask] + (max(flux)-flux)

# Plot Cleaned Spectra -------------------------------------------------------

plt.figure(dpi=200)

plt.axhline(1, color='black', alpha=0.8)
plt.plot(norm_wave[0], norm_flux[0], lw=1, alpha=0.7, \
         label=spec_labels[0], ds='steps-mid', color='blue')
plt.plot(norm_wave[1], norm_flux[1], lw=1, alpha=0.7, \
         label=spec_labels[1], ds='steps-mid', color='red')

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel(r'Normalized Flux')
plt.title('Removed Lines')

plt.xlim(wave_min, wave_max)
plt.ylim(bottom=-0.1, top=2)

plt.legend(fontsize=8)

plt.savefig((spec_dir+obj_name+'_cleaned.png'), format='png')

plt.show()

plt.clf()
