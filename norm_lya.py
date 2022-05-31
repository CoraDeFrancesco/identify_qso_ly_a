#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:25:42 2022

@author: user1
"""
##############################################################################
# Imports
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit

##############################################################################
# Functions
##############################################################################

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
    
    if interp_wave.any():
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

def norm_specs(waves, fluxes, norm_point=1700, width=1):
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

def eigen_fit(wave, *p):
    """
    Fit Bosman eigenvectors to a spectrum.

    Parameters
    ----------
    wave : arr
        Wavelength array of data. Must already be aligned with the wavelength array
        of red side Bosman eigenvectors.
    *p : arr (or array-like)
        Parameters of fit. Scaling coefficients of mean and eigenvectors.

    Returns
    -------
    Spectrum simulated from Bosman eigenvectors.

    """
    
    
    coeffs = p[:-1]
    shift=0
    
    z_guess = coeffs[-1]
    
    r_flux_vecs = align_r_flux[1:]
    
    curve = shift + np.exp(np.dot(coeffs, r_flux_vecs))
    
    return(curve)

##############################################################################
# Load Data, Eigenvecs
##############################################################################

wave_pca_r, pca_comp_r, wave_pca_b, \
    pca_comp_b,X_proj = pickle.load(open('pca_all_r15_b10_nn21_bosfit.pckl','rb'))
                                
# Look at PCA comps

plt.figure(dpi=200)

plt.plot(wave_pca_r, pca_comp_r[0], color='red', lw=3, label='R Mean', alpha=0.5)
plt.plot(wave_pca_b, pca_comp_b[0], color='blue', lw=3, label='B Mean', alpha=0.5)

for ri, pca_r in enumerate(pca_comp_r[1:]):
    plt.plot(wave_pca_r, pca_r, color='red', alpha=0.5)
for bi, pca_b in enumerate(pca_comp_b[1:]):
    plt.plot(wave_pca_b, pca_b, color='blue', alpha=0.5)
    
plt.legend()
    
plt.xlabel('Wavelength (A)')
plt.ylabel('Flux (unitless)')
plt.title('Bosman+21b PCA Eignevectors')
    
plt.show()
plt.clf()


# Bounds for the coefficients on the red side: safely inflated from 
# the training set. Best-fit values outside of these are quite suspicious!

# 15 values -> only the eigenvecs, not the mean!

bounds = [(0, 120), # bound for the mean - might realize this is a bad idea later lol
         (-60, 60),
         (-15, 15),
         (-8, 8),
         (-7, 7),
         (-5, 5),
         (-4, 4),
         (-3, 3),
         (-4, 4),
         (-2, 2),
         (-2, 2),
         (-2, 2),
         (-2, 2),
         (-2, 2),
         (-2, 2),         
         (-2, 2)]

# extra parameter for redshift "wiggle room", which is also part of our 
# method (values typical for redshift error at z=6-8)

bounds.append((-0.1,0.05))

# test spectrum

obj_name = 'J085825'
spec_dir = 'specs/J085825/' # data directory (with /)
spec_file_1 = 'spec-0468-51912-0036-dered.txt' # spectrum 1 file name
spec_file_2 = 'spec-3815-55537-0910.dr9' # spectrum 2 file name
spec_mjd_1 = '51912' # MJD of spectrum 1
spec_mjd_2 = '55537' # MjD of spectrum 2

z_test = 2.8684 # redshift of object (float)

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

# Bosman var names
lam = spec0[0] / (1+z_test)
flux = spec0[1]
err = spec0[2]

plt.figure(dpi=200)
plt.plot(lam, flux)

plt.xlabel('Wavelength (A)')
plt.ylabel('Flux (cgs)')
plt.title((obj_name+ ' '+ spec_mjd_1))

plt.show()
plt.clf()

norm_point = 1290

# Scale data flux by val at 1290 (Davies+ 2018)
lams, fluxes = norm_specs([spec0[0]/ (1+z_test), spec1[0]/ (1+z_test)], \
                       [spec0[1], spec1[1]], norm_point=norm_point)

lam = lams[0]
flux = fluxes[0]

plt.figure(dpi=200)
plt.plot(lam, flux)

plt.xlabel('Wavelength (A)')
plt.ylabel('Scaled Flux')
plt.title((obj_name+ ' '+ spec_mjd_1))
plt.axhline(1, color='gray', ls='--')
plt.axvline(norm_point, color='gray', ls='--')

plt.show()
plt.clf()

#

##############################################################################
# Align
##############################################################################

align_waves = [lam]
for i in range(0, len(pca_comp_r)):
    align_waves.append(wave_pca_r)
align_waves = np.asarray(align_waves)
align_waves = np.asarray(align_waves)
align_fluxes = [flux]
for i, comp in enumerate(pca_comp_r):
    align_fluxes.append(comp)
align_fluxes = np.asarray(align_fluxes)

wave_min = min(wave_pca_r)
wave_max = max(wave_pca_r)

align_err = []
for i, al_flux in enumerate(align_fluxes):
    zero_err = np.zeros(len(al_flux))
    align_err.append(zero_err)

align_r_wave, align_r_flux, align_r_err = align_data(align_waves, align_fluxes, \
               align_err, data_names=None, wave_min=wave_min,\
               wave_max=wave_max, get_res=True, save=False, save_dir=None, res=None,\
               interp_wave=lam)
    
interp_pca_r = align_r_flux[1:]
align_mean = align_r_flux[1]


plt.figure(dpi=200)

plt.plot(wave_pca_r, pca_comp_r[0], color='red', lw=2, label='R Mean', alpha=0.5)

for ri, pca_r in enumerate(pca_comp_r[1:]):
    plt.plot(wave_pca_r, pca_r, color='red', alpha=0.5)

for i, al_flux in enumerate(align_r_flux[1:]):
    
    plt.plot(lam, al_flux, color='purple', alpha=0.3, lw=4)

plt.xlim(1400, 1700)
plt.ylim(-0.2, 0.2)
    
plt.xlabel('Wavelength (A)')
plt.ylabel('Flux (unitless)')
    
plt.title('Aligned')

plt.show()
plt.clf()

##############################################################################
# Fit & Project
##############################################################################

shift_guess = 0
comps_guess = np.zeros(len(bounds))

# p0 = np.append(shift_guess, comps_guess)

p0 = comps_guess

popt, pcov = curve_fit(eigen_fit, lam, flux, p0=p0)


shift_fit=0

coeffs_fit = popt[:-1]

coeffs_r_q = coeffs_fit
coeffs_b_q = np.dot(coeffs_r_q, X_proj)
pca_q_r_10 = np.exp(np.dot(coeffs_r_q,pca_comp_r))
pca_q_b_10 = np.exp(np.dot(coeffs_b_q,pca_comp_b))

# Plot fit

plt.figure(dpi=200)

plt.plot(lam, flux, color='gray')
plt.plot(wave_pca_r, shift_fit+pca_q_r_10, color='red')
plt.plot(wave_pca_b, shift_fit+pca_q_b_10, color='blue')

plt.ylim(bottom=-0.01)
plt.xlim(min(wave_pca_b), max(wave_pca_r))

plt.xlabel('Wavelength (A)')
plt.ylabel('Scaled Flux')
plt.title(('Fit Result '+obj_name+ ' '+ spec_mjd_1))

plt.show()
plt.clf()

# Plot Ly-a forest region

plt.figure(dpi=200)

plt.plot(lam, flux, color='black')
plt.plot(wave_pca_r, shift_fit+pca_q_r_10, color='red')
plt.plot(wave_pca_b, shift_fit+pca_q_b_10, color='blue')
plt.axhline(1, color='gray', alpha=0.5, ls='--')

plt.ylim(bottom=-0.01)
plt.xlim(right=1216, left=min(wave_pca_b))

plt.xlabel('Wavelength (A)')
plt.ylabel('Scaled Flux')
plt.title(('Blue-Side Projection '+obj_name+ ' '+ spec_mjd_1))

plt.show()
plt.clf()

##############################################################################
# Normalize
##############################################################################



