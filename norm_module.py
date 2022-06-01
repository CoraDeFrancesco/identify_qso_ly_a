#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 08:28:14 2022

@author: user1
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit

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
        Parameters of fit. Scaling coefficients of mean (1), eigenvectors (15),
        z correction (1).

    Returns
    -------
    Spectrum simulated from Bosman eigenvectors.

    """
    coeffs = p[:-1]
    
    z_guess = coeffs[-1]
    
    r_flux_vecs = align_r_flux[1:]
    
    curve = np.exp(np.dot(coeffs, r_flux_vecs))
    
    return(curve)

def norm(data_wave, data_flux, norm_point=1290, plot_checks=True, plot_res=True):
    """
    Normalize using Bosman/Davies method with eigenvecotrs from Bosman+21b.

    Parameters
    ----------
    data_wave : arr
        Restframe wavelength array for data.
    data_flux : arr
        Flux for data.
    norm_point : float, optional
        Wavelength at which to scale data before normalization.
        The default is 1290 (from Davies+2018).
    plot_checks : bool, optional
        Display intermediate plots to check process. The default is True.
    plot_res : bool, optional
        Generate plots of fit and normalization. The default is True.

    Returns
    -------
    Normalized spectrum. Uses red side fit lam>1220A, blue side projection
    lam<=1220A.

    """
    # data_wave must be z corrected already!!!
    # we don't use the bounds yet... but we'll code them in here 
    #   so that I think about fixing that later.
    
    # Load Bosman means, eigenvecs, proj matrix
    
    wave_pca_r, pca_comp_r, wave_pca_b, \
        pca_comp_b,X_proj = pickle.load(open('pca_all_r15_b10_nn21_bosfit.pckl','rb'))
        
    bounds = [(0, 120), # bound for the mean 
         (-60, 60), # bounds for eigenvecs
         (-15, 15), # .
         (-8, 8),   # .
         (-7, 7),   # .
         (-5, 5),   # .
         (-4, 4),   # .
         (-3, 3),   # .
         (-4, 4),   # .
         (-2, 2),   # .
         (-2, 2),   # .
         (-2, 2),   # .
         (-2, 2),   # .
         (-2, 2),   # .
         (-2, 2),   # .   
         (-2, 2)]   # .

    # extra parameter for redshift "wiggle room", which is also part of our 
    # method (values typical for redshift error at z=6-8)
    
    bounds.append((-0.1,0.05))
    
    lam, flux = norm_specs([data_wave, data_wave], \
                       [data_flux, data_flux], norm_point=norm_point)
    lam=lam[0]
    flux = flux[0]
    
    
    if plot_checks: # Plot the data scaled to norm_point
        
        plt.figure(dpi=200)
        plt.plot(lam, flux)

        plt.xlabel('Wavelength (A)')
        plt.ylabel('Scaled Flux')
        plt.title(('Data Scaled to Unity at '+str(norm_point)))
        plt.axhline(1, color='gray', ls='--')
        plt.axvline(norm_point, color='gray', ls='--')
        
        plt.show()
        plt.clf()
    
    # Align Bosman red eigenvecs to wavelength resolution of data
    
    align_waves = [lam]
    for i in range(0, len(pca_comp_r)):
        align_waves.append(wave_pca_r)
    align_waves = np.asarray(align_waves)
    # align_waves = np.asarray(align_waves)
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
    
    # Function doing the aligning
    global align_r_flux # make sure we can call align_r_flux from model
    
    align_r_wave, align_r_flux, align_r_err = align_data(align_waves, align_fluxes, \
                   align_err, data_names=None, wave_min=wave_min,\
                   wave_max=wave_max, get_res=True, save=False, save_dir=None, res=None,\
                   interp_wave=lam)
        
    interp_pca_r = align_r_flux[1:]
    align_mean = align_r_flux[1]
    
    if plot_checks: # Plot aligned eigenvecs vs original
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
        
    # Fit eigenvecs to data
    
    comps_guess = np.zeros(len(bounds)) # guess all zeros for coeffs
    
    p0 = comps_guess # initial guess
    
    popt, pcov = curve_fit(eigen_fit, lam, flux, p0=p0) # fit
    
    coeffs_fit = popt[:-1] # results (ignore z correction)
    
    coeffs_r_q = coeffs_fit # red coefficients
    coeffs_b_q = np.dot(coeffs_r_q, X_proj) # project to blue coeffs
    pca_q_r_10 = np.exp(np.dot(coeffs_r_q,pca_comp_r)) # red total spec
    pca_q_b_10 = np.exp(np.dot(coeffs_b_q,pca_comp_b)) # blue total spec
    
    # Plot fit
    
    if plot_res:
        plt.figure(dpi=200)
        
        plt.plot(lam, flux, color='gray')
        plt.plot(wave_pca_r, pca_q_r_10, color='red')
        plt.plot(wave_pca_b, pca_q_b_10, color='blue')
        plt.axvline(1240)
        
        plt.ylim(bottom=-0.01)
        plt.xlim(min(wave_pca_b), max(wave_pca_r))
        
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Scaled Flux')
        plt.title(('Fit Result'))
        
        plt.show()
        plt.clf()
    
    # Plot Ly-a forest region
    
    if plot_checks:
        plt.figure(dpi=200)
        
        plt.plot(lam, flux, color='black')
        plt.plot(wave_pca_r, pca_q_r_10, color='red')
        plt.plot(wave_pca_b, pca_q_b_10, color='blue')
        plt.axhline(1, color='gray', alpha=0.5, ls='--')
        
        plt.ylim(bottom=-0.01)
        plt.xlim(right=1216, left=min(wave_pca_b))
        
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Scaled Flux')
        plt.title(('Blue-Side Projection'))
        
        plt.show()
        plt.clf()
        
    # Normalize
    
    # Adjust blue side to wavelenght resolution
    align_waves = [lam, wave_pca_b]
    align_fluxes = [flux, pca_q_b_10]
    
    wave_min = min(wave_pca_b)
    wave_max = max(wave_pca_b)
    
    align_err = []
    for i, al_flux in enumerate(align_fluxes):
        zero_err = np.zeros(len(al_flux))
        align_err.append(zero_err)
    
    align_b_wave, align_b_flux, align_b_err = align_data(align_waves, align_fluxes, \
                   align_err, data_names=None, wave_min=wave_min,\
                   wave_max=wave_max, get_res=True, save=False, save_dir=None, res=None,\
                   interp_wave=lam)
        
    align_pca_b = align_b_flux[1]
    
    # Adjust red side to wavelenght resolution
    align_waves = [lam, wave_pca_r]
    align_fluxes = [flux, pca_q_r_10]
    
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
        
    align_pca_r = align_r_flux[1]
    
    # Check Norm
    if plot_res:     
        plt.figure(dpi=200)
        
        plt.plot(lam, flux, color='black', alpha=0.5, label='Data')
        plt.plot(align_b_wave, align_pca_b, color='blue', label='Bos/Dav Fit')
        plt.axhline(1, color='black', ls='--', alpha=0.5)
        plt.plot(lam, flux/align_pca_b, color='green', label='Norm', alpha=0.8)
        
        plt.legend(fontsize=8)
        
        plt.xlim(min(wave_pca_b), max(wave_pca_b))
        plt.ylim(-0.01, 3)
        
        plt.xlabel('Wavelength (A)')
        plt.ylabel('Scaled Flux')
        plt.title(('Blue-Side Normalization'))
        
        plt.show()
        plt.clf()
        
    # Norm by red > 1220 A, blue <= 1220 A
    
    red_mask = np.where(lam > 1220)
    blue_mask = np.where(lam <= 1220)
    
    r_div = flux/align_pca_r
    b_div = flux/align_pca_b
    
    r_out = r_div[red_mask]
    b_out = b_div[blue_mask]
    
    spec_out = np.append(b_out, r_out)
    
        
    return(spec_out)

