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
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.optimize import curve_fit
import norm_module as nm
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev, make_interp_spline, splrep
from scipy.optimize import minimize

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


catch_negs = True # Adjust negative flux values to zero.

delta = 10 # Number of wavelength bins to fit in one gaussian abs line
perc_cut=0.80 # Between 0 and 1. Percentage of normalized flux abs must exceed.
wave_min=1060 # Min wavelength for finding peaks.
wave_max=1200 # Maximum wavelength for finding peaks
width=1 # Number of wavelength bins for a minimum to be considered a peak.
wmax=200 #Maximum width of abs lines in wavelength bins. The default is 5.
distance=2 #Miminum distance in wavelength bins between potential abs lines.
n_prom = 2 #how many times the noise level a peak must be
spl_smooth = 2 #spline smoothing coefficient
spl_deg = 3 #spline degree

#-----------------------------------------------------------------------------
#%% Functions
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
    
    #TODO: add option to adjust negative flux values to zero.
    
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
    # peaks, properties = find_peaks(x, height=(border, 0), \
    #                                distance=distance, width=(width, wmax), \
    #                                prominence=(prom))
    peaks, properties = find_peaks(x, height=(border, 0), \
                                   width=(width, wmax), \
                                   prominence=(prom))
    
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
    
    waves = np.array([1033, 1215, 1240.81, 1399.8, 1549, 1908, 2799])
    species = ['OVI', r'Ly$\alpha$', 'NV', 'SiIV+OIV', 'CIV', 'CIII', 'MgII']
    
    wl_range_mask = np.where((wave_min < waves) & (wave_max > waves))
    
    for i in wl_range_mask[0]:
        plt.axvline(waves[i], label=species[i], ls='--', color=('C'+str(i)))

# class fitFuncs:
    
#     def __init__(self):
        
#         pass

#     def gauss(wave, *p):
        
#         norm, mu, sigma, shift = p
        
#         curve = shift+norm*np.exp(-(wave-mu)**2/(2.*sigma**2))
        
#         return(curve)
    
#     def gauss_spl_conv(wave, *p):
        
#         norm, mu, sigma, shift = p
        
#         gauss_curve = shift+norm*np.exp(-(wave-mu)**2/(2.*sigma**2))
#         spl_curve = spl_spec(wave)
#         curve = gauss_curve * spl_curve
        
#         return(curve)
    
# models = fitFuncs()

def gauss_spl_conv(wave, *p):
    
    # norm, mu, sigma, shift = p
    norm, mu, sigma = p
    shift = 1
    
    gauss_curve = shift+norm*np.exp(-(wave-mu)**2/(2.*sigma**2))
    spl_curve = spl_spec
    curve = gauss_curve * spl_curve
    
    return(curve)

def gauss(wave, *p):
    
    # norm, mu, sigma, shift = p
    norm, mu, sigma = p
    shift = 1
    
    gauss_curve = shift+norm*np.exp(-(wave-mu)**2/(2.*sigma**2))

    return(gauss_curve)

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
            #print('     Options:', flat_idxs)
            # go to subsequent index lists
            matches = 0
            for fidx in flat_idxs:
                
                if fidx in range(idx_min, idx_max):
                    #print('     Found:', fidx)
                    matches += 1
            #print('     Total matches:', matches)
            if matches >= len(line_idxs):
                matched_list_i.append(idx)
                #print('     Adding idx to matched list.')
            #else:
                #print('     Not enough matches. Disregarding idx.')
        line_idx_matched.append(matched_list_i)
    
    return(line_idx_matched)

def get_noise(data_wave, data_flux, wave_min, wave_max):
    
    reg_mask = np.where((data_wave < wave_max) & (data_wave > wave_min))
    reg_flux = data_flux[reg_mask]
    noise=np.std(reg_flux)
    
    return(noise)

def split_spec(len_spec, len_seg):
    
    segs_idxs = []
    
    n_segs = int(len_spec / len_seg)
    for n in range(1, n_segs+1):
        seg_start = (n-1)*len_seg
        seg_end = (n*len_seg) -1
        new_seg = np.linspace(seg_start, seg_end, len_seg)
        segs_idxs.append(new_seg.astype(int))

    return(segs_idxs)

def find_segs_cont(wave, flux, npix=20, s=1000, k=3, noise_min=1125, noise_max=1175):
    
    #TODO: make sure to return segments covering the entire wavelength range provided.
    spec_split = split_spec(len(wave), npix)

    # Fit continuous cubic spline to each segment
    
    spl_fits1 = []
    
    
    for i, segment in enumerate(spec_split):
        
        # print('segment', i+1, 'of', len(spec_split))
        spl1 = UnivariateSpline(wave[segment], flux[segment], s=s, k=k)
        spl_fits1.append(spl1)
        
    plt.figure(dpi=200)
    plt.plot(wave, flux, alpha=0.5)
    for i, segment in enumerate(spec_split):
        plt.plot(wave[segment], spl_fits1[i](wave[segment]), color='blue')
    plt.ylim(-0.01, 2.1)
    plt.xlim(wave_min, wave_max)
    plt.show()
    plt.clf()
    
    # Reject piexels in each segment which lie more than two sigma below the fit
    
    for i, segment in enumerate(spec_split): # refit one segment at a time
    
    
        seg_noise = np.std(flux[segment])
        spec_noise = get_noise(wave, flux, noise_min, noise_max)
        # print('spec_noise = ', spec_noise)
    
        segment_ini = segment
        
        spl_flux = spl_fits1[i](wave[segment])
        seg_flux = flux[segment]
        seg_wave = wave[segment]
        sigma_flux = np.std(seg_flux)
        
        flux_dists = seg_flux - spl_flux
        flux_dists_sigma = flux_dists / sigma_flux

        big_sig_mask = np.where(flux_dists <= -spec_noise)

        big_counter = len(big_sig_mask[0])
        while big_counter > 0:

            segment = np.delete(segment, big_sig_mask[0])

            seg_flux = flux[segment]
            seg_wave = wave[segment]
            
            spl_refit = UnivariateSpline(seg_wave, seg_flux, s=s, k=k)
            
            spl_fits1[i] = spl_refit
            
            seg_flux = flux[segment]
            spl_flux = spl_refit(wave[segment])
            sigma_flux = np.std(seg_flux)
            
            flux_dists = seg_flux - spl_flux
            flux_dists_sigma = flux_dists / sigma_flux
            
            # big_sig_mask = np.where(flux_dists_sigma <= -2.0)
            big_sig_mask = np.where(flux_dists <= -spec_noise)
            big_counter = len(big_sig_mask[0])
            
    plt.figure(dpi=200)
    plt.plot(wave, flux, alpha=0.5)
    for i, segment in enumerate(spec_split):
        plt.plot(wave[segment], spl_fits1[i](wave[segment]), color='blue')
    plt.ylim(-0.01, 2.1)
    plt.xlim(wave_min, wave_max)
    plt.show()
    plt.clf()     
    
    spl_model_out = np.array([])
    for i, segment in enumerate(spec_split):
        spl_model_out = np.concatenate((spl_model_out, spl_fits1[i](wave[segment])))
    diff_len = len(wave) - len(spl_model_out)
    spl_model_out = np.concatenate((spl_model_out, np.ones(diff_len)))

    return(spl_model_out)

def guess(x, y, k, s, w=None):
    """Do an ordinary spline fit to provide knots (credit:@askewchan)"""
    return splrep(x, y, w, k=k, s=s)

def err(c, x, y, t, k, w=None):
    """The error function to minimize (credit:@askewchan)"""
    diff = y - splev(x, (t, c, k))
    if w is None:
        diff = np.einsum('...i,...i', diff, diff)
    else:
        diff = np.dot(diff*diff, w)
    return np.abs(diff)

def spline_neumann(x, y, k=3, s=0, w=None, anchor=None):
    '''
    Spline fitting with smoothing and boundary conditions. (credit:@askewchan)

    Parameters
    ----------
    x : arr
        wavelength array.
    y : arr
        flux array.
    k : int, optional
        Degree of polynomial spline. The default is 3.
    s : float, optional
        Smoothing factor. The default is 0.
    w : arr, optional
        Wights. The default is None.
    anchor : float, optional
        Anchor flux value from previous segment.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
        
    t, c0, k = guess(x, y, k, s, w=w)
    x0 = x[0] # point at which zero slope is required (first point)
    x_end = x[-1] #also require zero slope at end
    if anchor:
        print('anchoring at', anchor)
        # con = ({'type': 'eq', 'fun': lambda c: x0 - anchor}, \
        #        {'type': 'eq', \
        #         'fun': lambda c: splev([x_end], (t, c, k), der=2)}, \
        #        {'type': 'eq', \
        #         'fun': lambda c: splev([x0], (t, c, k), der=1)})
        con = ({'type': 'eq', 'fun': lambda c: x0 - anchor})
    else:
        print('not anchoring')
        con = ({'type': 'eq', \
               'fun': lambda c: splev([x_end], (t, c, k), der=2)}, \
               {'type': 'eq', \
                'fun': lambda c: splev([x0], (t, c, k), der=1)})
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

def find_segs_cont_bound(wave, flux, npix=16, s=0.5, k=3, \
                         noise_min=1125, noise_max=1175, sig_cut=1.5):
    
    spec_split = split_spec(len(wave), npix)

    # Fit continuous cubic spline to each segment
    
    spl_fits1 = []
    
    for i, segment in enumerate(spec_split):
        
        print('WORKING ON SEGMENT', i, 'FIRST PASS')
        
        # sp0 = UnivariateSpline(wave[segment], flux[segment], k=k, s=s)
        if i==0:
            anchor=None
            print('   anchor set to', anchor)
        # elif ((spl_fits1[i-1](wave[spec_split[i-1]]))[-1] \
        #       > 1+ sig_cut*get_noise(wave, flux, noise_min, noise_max)):
        #     anchor=1
        #     print('   anchor set to', anchor)
        else:
            anchor=(spl_fits1[i-1](wave[spec_split[i-1]]))[-1]
            if np.isnan(anchor):
                print('NAN IDENTIFIED')
                anchor=1
            print('   anchor set to', anchor)
        spl1 = spline_neumann(wave[segment], flux[segment], k=k, s=s, anchor=anchor)
        
        spl_fits1.append(spl1)
        
    plt.figure(dpi=200)
    plt.title('Spline Guess')
    plt.plot(wave, flux, alpha=0.5)
    # plt.plot(x, conv_2, alpha=0.5)
    for i, segment in enumerate(spec_split):
        plt.plot(wave[segment], spl_fits1[i](wave[segment]), color='blue')
    plt.ylim(-0.01, 2.1)
    plt.xlim(1060, 1200)
    plt.ylabel('Normalized Flux')
    plt.xlabel('Rest Frame Wavelength (A)')
    plt.show()
    plt.clf()
    
    # Reject piexels in each segment which lie more than 1.5 sigma below the fit
    
    for i, segment in enumerate(spec_split): # refit one segment at a time
        
        print('WORKING ON SEGMENT', i, 'REFIT')
        
        if i==0:
            anchor=None
            print('   anchor set to none')
        # elif ((spl_fits1[i-1](wave[spec_split[i-1]]))[-1] \
        #       > 1+ sig_cut*get_noise(wave, flux, noise_min, noise_max)):
        #     anchor=1
        #     print('   anchor set to', anchor)
        else:
            anchor=(spl_fits1[i-1](wave[spec_split[i-1]]))[-1]
            if np.isnan(anchor):
                print('NAN IDENTIFIED')
                anchor=1
            print('   anchor set to', anchor)
    
        # seg_noise = np.std(flux[segment])
        spec_noise = get_noise(wave, flux, noise_min, noise_max)
        
    
        # segment_ini = segment
        
        spl_flux = spl_fits1[i](wave[segment])
        seg_flux = flux[segment]
        seg_wave = wave[segment]
        sigma_flux = np.std(seg_flux)
        # print(sigma_flux)
        
        flux_dists = seg_flux - spl_flux
        flux_dists_sigma = flux_dists / sigma_flux
        
        
        big_sig_mask = np.where(flux_dists <= -spec_noise)
        # big_sig_mask = np.where(np.abs(flux_dists_sigma) >= sig_cut) #flag both
                                        # above and below n sigma
        
        
        big_counter = len(big_sig_mask[0])
        while big_counter > 0:
            
            segment = np.delete(segment, big_sig_mask[0])
            
            seg_flux = flux[segment]
            seg_wave = wave[segment]
            
            # sp0_refit = UnivariateSpline(wave[segment], flux[segment], k=k, s=s)
            # spl1 = spline_neumann(wave[segment], flux[segment], k=k, s=s)
            spl_refit = spline_neumann(seg_wave, seg_flux, s=s, k=k, anchor=anchor)
            
            spl_fits1[i] = spl_refit
            
            spl_flux = spl_refit(wave[segment])
            
            sigma_flux = np.std(seg_flux)
            
            flux_dists = seg_flux - spl_flux
            flux_dists_sigma = flux_dists / sigma_flux
            
            big_sig_mask = np.where(flux_dists_sigma <= -sig_cut)
            # big_sig_mask = np.where(np.abs(flux_dists_sigma) >= sig_cut) #flag both
                                        # above and below n sigma
            big_counter = len(big_sig_mask[0])
            
    plt.figure(dpi=200)
    plt.title('Spline Fit')
    plt.plot(wave, flux, alpha=0.5)
    # plt.plot(x, conv_2, alpha=0.5)
    for i, segment in enumerate(spec_split):
        plt.plot(wave[segment], spl_fits1[i](wave[segment]), color='blue')
    plt.ylim(-0.01, 2.1)
    plt.xlim(1060, 1200)
    plt.ylabel('Normalized Flux')
    plt.xlabel('Rest Frame Wavelength (A)')
    plt.show()
    plt.clf()     
    
    
    spl_model_out = np.array([])
    for i, segment in enumerate(spec_split):
        spl_model_out = np.concatenate((spl_model_out, spl_fits1[i](wave[segment])))
        
    len_ones = len(wave) - len(spl_model_out)
    full_model_out = np.concatenate((spl_model_out, np.ones(len_ones)))

    return(full_model_out)

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
                       data_names=data_names, wave_min=wave_min, \
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
 
#%% Good normalization ---------------------------------------------------------

norm_wave = []
norm_flux = []

for i, align_flux in enumerate(align_fluxes):
    
    bos_norm = nm.norm(align_wave, align_flux, plot_checks=False)
    norm_wave.append(align_wave)
    print(bos_norm)
    if catch_negs:
        neg_mask = np.where(bos_norm < 0)
        bos_norm[neg_mask] = 0.0
    norm_flux.append(bos_norm)


# Check normalization --------------------------------------------------------

plt.figure(dpi=200)

plt.axhline(1, color='black', alpha=0.8)
plt.plot(norm_wave[0], norm_flux[0], lw=1, alpha=0.7, \
         label=spec_labels[0], ds='steps-mid', color='blue')
plt.plot(norm_wave[1], norm_flux[1], lw=1, alpha=0.7, \
         label=spec_labels[1], ds='steps-mid', color='red')

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel('Normalized Flux')
plt.title('Normalized Spectra - Ly-a Zoom')

plt.xlim(wave_min, wave_max)
plt.ylim(bottom=-0.1, top=2)

plt.legend(fontsize=8)

plt.show()

plt.clf()


# Plot whole normalized specs

plt.figure(dpi=200)

# plt.axhline(1, color='black', alpha=0.8)
plot_sdss_lines(900, 1700)
plt.plot(norm_wave[0], norm_flux[0], lw=1, alpha=0.7, \
         label=spec_labels[0], ds='steps-mid', color='blue')
plt.plot(norm_wave[1], norm_flux[1], lw=1, alpha=0.7, \
         label=spec_labels[1], ds='steps-mid', color='red')
# plt.plot(norm_point, 1, color='orange', marker='*', label='Scale Point', ls='')

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel('Scaled Flux')
plt.title('Normalized Spectra')

plt.xlim(1000, 1700)
plt.ylim(bottom=-0.1)

plt.legend(fontsize=8)

plt.show()

plt.clf()

#%% Find Noise -----------------------------------------------------------------

prom1 = n_prom*get_noise(norm_wave[0], norm_flux[0], 1600, 1800)
prom2 = n_prom*get_noise(norm_wave[1], norm_flux[1], 1600, 1800)

#%% Find Peaks -----------------------------------------------------------------

line_waves1, line_fluxes1, line_idxs1 = get_line_mins(norm_wave[0], norm_flux[0], \
                                        perc_cut=perc_cut, wave_min=wave_min, \
                                        wave_max=wave_max, width=width, wmax=wmax,\
                                        plot=True, prom=prom1)
    
line_waves2, line_fluxes2, line_idxs2 = get_line_mins(norm_wave[1], norm_flux[1], \
                                        perc_cut=perc_cut, wave_min=wave_min, \
                                        wave_max=wave_max, width=width, wmax=wmax,\
                                        plot=True, prom=prom2)
    
line_waves = [line_waves1, line_waves2]
line_fluxes = [line_fluxes1, line_fluxes2]
line_idxs = [line_idxs1, line_idxs2]

#%% Match Lines ----------------------------------------------------------------

line_idx_matched = match_lines(line_idxs, distance)
line_waves_matched = [norm_wave[0][line_idx_matched[0]], norm_wave[1][line_idx_matched[1]]]
line_fluxes_matched = [norm_flux[0][line_idx_matched[0]], norm_flux[1][line_idx_matched[1]]]

#line_idxs_save = [line_idx_matched, line_idx_matched]

# Plot Matches -------------------------------------------------------------

plt.figure(dpi=200)

# plt.axhline(1, color='black', alpha=0.8)
plot_sdss_lines(wave_min, wave_max)
plt.plot(norm_wave[0], norm_flux[0], lw=1, alpha=0.7, \
         label=spec_labels[0], ds='steps-mid', color='blue')
plt.plot(norm_wave[1], norm_flux[1], lw=1, alpha=0.7, \
         label=spec_labels[1], ds='steps-mid', color='red')
plt.plot(norm_wave[0][line_idx_matched[0]], norm_flux[0][line_idx_matched[0]], ls='', alpha=0.7, \
         label='Matched', color='blue', marker='*')
plt.plot(norm_wave[1][line_idx_matched[1]], norm_flux[1][line_idx_matched[1]], ls='', alpha=0.7, \
         color='red', marker='*')
# plt.plot(norm_point, 1, color='orange', marker='*', label='Scale Point', ls='')

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel('Scaled Flux')
plt.title('Normalized Spectra')

plt.xlim(wave_min, wave_max)
plt.ylim(bottom=-0.1, top=1.5)

plt.legend(fontsize=8)

plt.show()

plt.clf()

#%% Smooth specs, fit spline ---------------------------------------------------

wave_mask = np.where((norm_wave[0] >= wave_min) & (norm_wave[0] <= wave_max))
spl1 = find_segs_cont_bound(norm_wave[0], norm_flux[0], noise_min=1600, noise_max=1800)
spl2 = find_segs_cont_bound(norm_wave[1], norm_flux[1], noise_min=1600, noise_max=1800)
    
spl_funcs = [spl1, spl2]

plt.figure(dpi=200)
plt.plot(norm_wave[0], norm_flux[0], lw=1, alpha=0.2, \
          label=spec_labels[0], ds='steps-mid', color='blue')
plt.plot(norm_wave[1], norm_flux[1], lw=1, alpha=0.2, \
          label=spec_labels[1], ds='steps-mid', color='red')
plt.plot(norm_wave[0], spl1, color='green')
plt.plot(norm_wave[1], spl2, color='orange')

plt.xlim(wave_min, wave_max)
plt.ylim(0, 2)

plt.title('Spline Fit')

plt.show()
plt.clf()


#%% Fit Ly-a Lines -------------------------------------------------------------

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
    
    for i, line_idx in enumerate(line_idx_matched[spec_idx]):
        
        #line_wave = line_waves_matched[spec_idx][i]
        line_wave = norm_wave[spec_idx][line_idx]
        
        min_idx = line_idx-delta
        max_idx = line_idx+delta
        if min_idx <= 0:
            min_idx = 0
        elif max_idx >=  len(norm_wave[0]):
            max_idx = len(norm_wave[0])
        
        xdata = norm_wave[spec_idx][min_idx:max_idx]
        ydata = norm_flux[spec_idx][min_idx:max_idx]
        
        global spl_spec
        spl_spec = spl_funcs[spec_idx][min_idx:max_idx]
        
        # p0 = [-1, line_wave, 0.5, 1] # initial guess (norm, mu, sigma, shift)
        p0 = [-1, line_wave, 0.5] #must grow ly-a from 1
        bounds = ((-np.inf, line_wave-0.5, 0), (0, line_wave+0.5, 0.5))
        
        try:
            popt, pcov = curve_fit(gauss_spl_conv, xdata, ydata, p0=p0, bounds=bounds)
            fit_flux = gauss_spl_conv(xdata, *popt)
        
            fit_waves_spec.append(xdata)
            #fit_fluxes_spec.append(fit_flux)
            fit_fluxes_spec.append(gauss(xdata, *popt))
        except RuntimeError:
            print('RuntimeError for line', line_wave)
            err_waves_spec.append(line_wave)
            err_fluxes_spec.append(norm_flux[spec_idx][line_idx])
            
    err_waves_spec = np.asarray(err_waves_spec) 
    
    fit_waves.append(fit_waves_spec)
    fit_fluxes.append(fit_fluxes_spec)
    err_waves.append(err_waves_spec)
    err_fluxes.append(err_fluxes_spec)
    

#%% Remove lines -------------------------------------------------------------

# Plot Ly-a fits -------------------------------------------------------------
    
    
plt.figure(dpi=200)

plt.plot(norm_wave[0], norm_flux[0], alpha=0.7, color='blue', ds='steps-mid')
plt.plot(norm_wave[1], norm_flux[1], alpha=0.7, color='red', ds='steps-mid')
plt.plot(norm_wave[0][line_idx_matched[0]], norm_flux[0][line_idx_matched[0]], 'b*')
plt.plot(norm_wave[1][line_idx_matched[1]], norm_flux[1][line_idx_matched[1]], 'r*')
# plt.plot(norm_wave[0][spline_mask_1], spl1(norm_wave[0])[spline_mask_1], color='green')
# plt.plot(norm_wave[1][spline_mask_1], spl2(norm_wave[1])[spline_mask_1], color='orange')
plt.axhline(1, color='black', alpha=0.5, ls='--', label='Norm')
plt.axhline(perc_cut, color='red', alpha=0.5, ls='--', label='Percent Cutoff')
plt.plot(err_waves[0], err_fluxes[0], 'gx', label='Fit Error')
plt.plot(err_waves[1], err_fluxes[1], 'gx')

for spec_idx in [0,1]:
    
    # colors = ['blue', 'red']
    colors=['black', 'black']
    
    for i, flux in enumerate(fit_fluxes[spec_idx]):
        
        plt.plot(fit_waves[spec_idx][i], flux+1, 'r-', alpha=0.3, \
                 color=colors[spec_idx], lw=2)

plt.xlabel('Rest Frame Wavelength (A)')
plt.ylabel(r'Normalized Flux')
plt.title('Fit Ly-a Lines')
    
plt.legend(fontsize=8)

plt.xlim(wave_min, wave_max)
# plt.xlim(1080, 1140)
plt.ylim(-0.1, 2.1)

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
plt.ylim(bottom=-0.1, top=2.1)

plt.legend(fontsize=8)

plt.savefig((spec_dir+obj_name+'_cleaned.png'), format='png')

plt.show()

plt.clf()

# Plot Full Spectra ----------------------------------------------------------

# plt.figure(dpi=200)

# plt.axhline(1, color='black', alpha=0.8)
# plt.plot(norm_wave[0], norm_flux[0], lw=1, alpha=0.7, \
#          label=spec_labels[0], ds='steps-mid', color='blue')
# plt.plot(norm_wave[1], norm_flux[1], lw=1, alpha=0.7, \
#          label=spec_labels[1], ds='steps-mid', color='red')

# plt.xlabel('Rest Frame Wavelength (A)')
# plt.ylabel(r'Normalized Flux')
# plt.title('Removed Lines')

# plt.xlim(wave_min)
# plt.ylim(bottom=-0.1, top=2)

# plt.legend(fontsize=8)

# # plt.savefig((spec_dir+obj_name+'_cleaned.png'), format='png')

# plt.show()

# plt.clf()


