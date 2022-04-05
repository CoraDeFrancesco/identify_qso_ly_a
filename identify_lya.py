#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 17:36:11 2022

@author: CoraAnn
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
#from scipy.fft import fft, fftfreq

# From PRH
# sflux3=np.interp(wavelength2,wavelength,sflux,left=0,right=0)
# serror3=np.interp(wavelength2,wavelength,serror,left=0,right=0)

#%% Useful functions

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

def plot_sdss_lines(wave_min, wave_max):
    """
    Adds vertical lines marking some common SDSS spectral lines
    (Ly-a, NV, SiIV+OIV, CIV, CIII, MGII). Only plots withing the given 
    wavelength range.
    
    Labels are included. To display, please add 'plt.lenend()' to your script.

    Parameters
    ----------
    wave_min : float
        Minimum wavelength for plotting.
    wave_max : float
        Maximum wavelength for plotting.

    Returns
    -------
    None.

    """
    
    waves = np.array([1215, 1240.81, 1399.8, 1549, 1908, 2799])
    species = [r'Ly$\alpha$', 'NV', 'SiIV+OIV', 'CIV', 'CIII', 'MgII']
    
    wl_range_mask = np.where((wave_min < waves) & (wave_max > waves))
    
    for i in wl_range_mask[0]:
        plt.axvline(waves[i], label=species[i], ls='--', color=('C'+str(i)))

def divide_specs(spec1, spec2):
    """
    Divide aligned spectral fluxes.
    
    Both flux arrays must have the same dimension (1D array of length n).

    Parameters
    ----------
    spec1 : arr
        Flux values of spectrum 1.
    spec2 : arr
        Flux values of spectrum 2.

    Returns
    -------
    Divided flux array (spec1 / spec2).
    If both have a value of 0 at a point, returns 1.
    If one has a value of 0 at a point, 0 flux is reset to a small number.

    """
    
    div_flux_list = []
    
    for i,flux1 in enumerate(spec1):
        
        flux2 = spec2[i]
        
        div_val = 1000
        
        if ((flux1==0) and (flux2==0)):
            div_val = 1
        elif ((flux1==0) and (flux2!=0)):
            flux1 = 0.00000000000001
            #div_val = 0
            div_val = flux1/flux2
        elif ((flux1!=0) and (flux2==0)):
            flux2 = 0.00000000000001
            #div_val = 0
            div_val = flux1/flux2
        else:
            div_val = flux1/flux2
        
        div_flux_list.append(div_val)
        
    return(div_flux_list)

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
                  distance=3, width=2, perc_cut=1.0, plot=True):
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
        Miminum distance in wavelength bins between potential abs lines. The default is 3.
    width : float, optional
        Minimun width of abs lines in wavelength bins. The default is 2.
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
    peaks, properties = find_peaks(x, height=(border, 0), distance=distance, width=width)
    
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

#%% Compare specs - single object

spec_dir = 'specs/J085825/'
# spec_dir = 'specs/J014548/'

z = 2.863
# z = 2.8

obj_specs = []
obj_names = []
for spec in os.listdir(spec_dir):
    data = np.loadtxt((spec_dir+spec)).T
    obj_specs.append(data)
    obj_names.append(spec)
    
spec0 = obj_specs[0]
spec1 = obj_specs[1]

# Overplot

# Ok cool...
# Lots of little lines are in the same place on both :)
# IDK how to get gaussians there, but let me put some vertical
#   lines over the places where they match.

plt.figure(dpi=200)

# for i,spec in enumerate(obj_specs):
    
#     plt.plot(spec[0]/(1 + z), spec[1], lw=1, alpha=0.75, label=obj_names[i], ds='steps-mid')


plt.plot(spec0[0]/(1 + z), spec0[1], lw=1, alpha=0.7, \
         label='J085825 MJD: 51912', ds='steps-mid', color='blue')
plt.plot(spec1[0]/(1 + z), spec1[1], lw=1, alpha=0.7, \
         label='J085825 MJD: 55537', ds='steps-mid', color='red')
#plt.plot(spec[0]/(1 + z), spec[0], lw=1, alpha=0.75, label=obj_names[i], ds='steps-mid')

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel(r'Flux ($10^{-17}$ erg/cm$^2$/s/A)')

plt.xlim(1060, 1200)
plt.ylim(bottom=-0.1, top=25)

#plt.axvline(4033, alpha=0.4, color='black')
#plt.axvline(4150, alpha=0.4, color='black')


plt.legend(fontsize=8)    
plt.show()
plt.clf()



#%% Presentation plots

plt.figure(dpi=200)
spec = obj_specs[1]
plt.plot(spec[0]/(1 + z), spec[1], lw=1, label='J085825 MJD: 55537',\
         ds='steps-mid', color='black')
plot_sdss_lines(900, 1700)

plt.axvspan(ymin=-0.1, ymax=32, xmin=950, xmax=1216, color='blue', alpha=0.2)

plt.xlim(950, 1700)
plt.ylim(bottom=-0.1, top=32)

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel(r'Flux ($10^{-17}$ erg/cm$^2$/s/A)')

plt.legend(fontsize=8, loc='upper right')    
plt.show()
plt.clf()

#%% Divide specs

# First align

data_waves = [spec0[0]/(1 + z), spec1[0]/(1 + z)]
data_fluxes = [spec0[1], spec1[1]]
data_errs = [spec0[2], spec1[2]]
data_names = ['J085825 MJD: 51912', 'J085825 MJD: 55537']

align_wave, align_fluxes, align_errs =  align_data(data_waves, data_fluxes, data_errs, \
                       data_names=data_names, wave_min=800, \
                       wave_max=2300, get_res=True, save=False)

# Check

for i, flux in enumerate(align_fluxes):
    plt.plot(align_wave, flux, label=data_names[i], lw=1, ds='steps-mid')

    
plt.xlim(1060, 1200)
plt.ylim(bottom=-0.1, top=25)
plt.legend()
plt.title('Aligned Specs')
plt.show()
plt.clf()

# Divide

div_flux = divide_specs(align_fluxes[1], align_fluxes[0])

plt.plot(align_wave, div_flux, 'r-', ms=1.5)
plt.axhline(1)
 
plt.xlim(1060, 1200)
plt.ylim(bottom=0, top=2)
plt.legend()
plt.title('Divided Flux')
plt.show()
plt.clf()

#%% Crude normalization

for i, flux in enumerate(align_fluxes):
    plt.plot(align_wave, flux, label=data_names[i], lw=1, ds='steps-mid')

    
#plt.xlim(1060, 1200)
plt.ylim(bottom=-0.1, top=25)
plt.legend()
plt.title('Aligned Specs')
plt.show()
plt.clf()

norm_wave, norm_flux = norm_specs((align_wave, align_wave), align_fluxes)
    
for i, flux in enumerate(norm_flux):
    plt.plot(norm_wave[i], flux, label=data_names[i], lw=1, ds='steps-mid')

    
#plt.xlim(1060, 1200)
plt.ylim(bottom=-0.1, top=4)
plt.legend()
plt.title('Normed Specs')
plt.show()
plt.clf()

#%% Find some peaks!

line_waves, line_fluxes, line_idxs = get_line_mins(norm_wave[1], norm_flux[1], \
                                        perc_cut=0.75, wave_min=1060, wave_max=1200, width=2)

#%% Generalize to all lines

def gauss(wave, *p):
    
    norm, mu, sigma = p
    
    curve = 1+norm*np.exp(-(wave-mu)**2/(2.*sigma**2))
    
    return(curve)

delta = 4

fit_waves = []
fit_fluxes = []
err_waves = []
err_fluxes = []

for i, line_idx in enumerate(line_idxs):
    
    line_wave = line_waves[i]
    
    xdata = norm_wave[1][line_idx-delta:line_idx+delta]
    ydata = norm_flux[1][line_idx-delta:line_idx+delta]
    
    p0 = [-1, line_wave, 0.5]
    try:
        popt, pcov = curve_fit(gauss, xdata, ydata, p0=p0)
        fit_flux = gauss(xdata, *popt)
    
        fit_waves.append(xdata)
        fit_fluxes.append(fit_flux)
    except RuntimeError:
        print('RuntimeError for line', line_wave)
        err_waves.append(line_wave)
        err_fluxes.append(norm_flux[1][line_idx])
        
err_waves = np.asarray(err_waves) 
    
    
plt.figure(dpi=200)

plt.plot(norm_wave[0], norm_flux[0], alpha=0.6, color='green', ds='steps-mid')
plt.plot(norm_wave[1], norm_flux[1], alpha=0.6, color='blue', ds='steps-mid')
plt.plot(line_waves, line_fluxes, 'r*', label='Abs Mins')
plt.axhline(1, color='black', alpha=0.5, ls='--', label='Norm')
plt.axhline(0.75, color='red', alpha=0.5, ls='--', label='Percent Cutoff')
plt.plot(err_waves, err_fluxes, 'gx', label='Fit Error')

for i, flux in enumerate(fit_fluxes):
    
    plt.plot(fit_waves[i], flux, 'r-.', alpha=0.8, color='red')
    
plt.legend()

plt.xlim(1060, 1200)
plt.ylim(-0.1, 2)

plt.show()
plt.clf()

#%% Similarity Metric

plt.figure(dpi=100)

sim_cut = 0.025

sim_metric = (abs((norm_flux[0] - norm_flux[1]))**2)
sim_mask = np.where(sim_metric < sim_cut)

plt.axhline(sim_cut, color='red', alpha=0.5, label='Sim Cutoff')

plt.plot(norm_wave[0], sim_metric, color='black', ds='steps-mid', alpha=0.5, label='Sim Metric')
plt.plot(norm_wave[0], norm_flux[0], alpha=0.4, color='green', ds='steps-mid')
plt.plot(norm_wave[1], norm_flux[1], alpha=0.4, color='blue', ds='steps-mid')

# plt.plot(norm_wave[0][sim_mask], norm_flux[0][sim_mask], alpha=0.6, label='Similar', ds='steps-mid', color='red')
# plt.plot(norm_wave[1][sim_mask], norm_flux[1][sim_mask], alpha=0.6, color='red')

plt.legend()
plt.xlim(1060, 1200)
plt.ylim(-0.1, 2)

plt.show()
plt.clf()




#%% Play with FFT

# yf0 = fft(-1*norm_flux[0])
# yf1 = fft(-1*norm_flux[1])

# ms_diff = np.abs(((yf0 - yf1)/2) ** 2)

# #plt.plot(norm_wave[0], norm_flux[0], alpha=0.6, color='green', ds='steps-mid')
# plt.plot(norm_wave[0], yf0/20, ds='steps-mid')
# plt.plot(norm_wave[0], yf1/20, ds='steps-mid')
# #plt.plot(norm_wave[0], ms_diff/20, ds='steps-mid', color='black', ls='', marker='o')

# plt.ylim(-2, 2)
# plt.xlim(1060, 1200)

# plt.show()
# plt.clf()

# yshift0 = np.fft.fftshift(yf0)

# plt.plot(norm_wave[0], yshift0)
# plt.ylim(-20, 20)
# plt.xlim(1060, 1200)