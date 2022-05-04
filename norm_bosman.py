#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 12:58:19 2022

@author: user1
"""
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as op

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

# how to load wavelengths on both sides, components, and the 
# projection matrix from the pickle

import pickle
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

bounds = [(-60, 60),
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

plt.figure(dpi=200)
plt.plot(lam, flux)

plt.xlabel('Wavelength (A)')
plt.ylabel('Flux (cgs)')
plt.title((obj_name+ ' '+ spec_mjd_1))

plt.show()
plt.clf()

# Align PCA eigenvecs and data in wavelength bins

align_waves = np.append(pca_comp_r, lam)
align_fluxes = np.append(wave_pca_r, flux)
wave_min = min(wave_pca_r)
wave_max = max(wave_pca_r)
align_err = np.zeros(shape=align_waves.shape)

align_r_vecs = align_data(align_waves, align_fluxes, align_err, data_names=None, wave_min=wave_min,\
               wave_max=wave_max, get_res=True, save=False, save_dir=None, res=None,\
               interp_wave=lam)
    
    
#%%


# define some likelihood that you want to minimize, then that 
# becomes a chi-squared (feel free to experiment)

def lnlike_q(theta):
    z = z_test+theta[-1] # add the z wiggle to redshift
    
    coeffs = np.append(1.0,theta[:-1]) # coefficients of eigenvecs with coeff for mean (1)
    
    
    C_dec = np.exp(np.dot(np.append(1.0,theta[:-1]),interp_pca_r(lam)))  # lam is wavelength
    
    chi2 = ivar_q_fit*np.power(flux-C_dec,2.0)
    
    return -np.sum(chi2)

# chi2_q = lambda *args: -2 * lnprob_q(*args)
chi2_q = lambda *args: -2 * lnlike_q(*args)


# Initial guess:  is the mean quasar spectrum

n_comp_r = pca_comp_r.shape[0] # number of red PCA eigenvectors
guess = np.zeros(n_comp_r) # guess zeros for all coefficients


# two example scipy-optimize methods that you may want to try:


result_q = op.minimize(chi2_q, guess, bounds = bounds)

result_q = op.differential_evolution(chi2_q, bounds = bounds,popsize=30,recombination=0.5,polish=True, disp=True, tol=0.02)


# one way of getting the coefficients out and projecting them:

dz_q = result_q.x[-1]
coeffs_r_q = np.append(1.0,result_q.x[:-1])
coeffs_b_q = np.dot(coeffs_r_q,X_proj)
pca_q_r_10 = np.exp(np.dot(coeffs_r_q,pca_comp_r))
pca_q_b_10 = np.exp(np.dot(coeffs_b_q,pca_comp_b))
wave10_r = wave_pca_r*(1+z_test+dz_q)/(1+z_test)
wave10_b = wave_pca_b*(1+z_test+dz_q)/(1+z_test)



