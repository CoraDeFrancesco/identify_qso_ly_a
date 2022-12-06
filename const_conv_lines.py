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
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev, make_interp_spline, splrep
from scipy.optimize import minimize

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

#%% Functions ------------------------------------------------------------------

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



#%% Get peaks, match


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

# lya_masks = []

# for line_list in line_idx_matched:
#     lya_mask = np.asarray([])
#     for idx in line_list:
#         min_val = idx - delta
#         max_val = idx + delta
#         idxs = np.linspace(min_val, max_val, (2*delta)+1)
#         lya_mask = np.concatenate((lya_mask, idxs))
#     lya_mask = np.unique(lya_mask)
#     lya_masks.append(lya_mask)

# spline_mask_1 = []
# spline_mask_2 = []
# for i in range(0, len(x)):
#     if i not in lya_masks[0]:
#         spline_mask_1.append(i)
#     if i not in lya_masks[1]:
#         spline_mask_2.append(i)

        

# #Spline fitting with smoothing s and degree k
# spl1 = UnivariateSpline(x[spline_mask_1], conv_1[spline_mask_1], s=1, k=3)
# spl2 = UnivariateSpline(x[spline_mask_2], conv_2[spline_mask_2], s=1, k=3)

# plt.figure(dpi=200)
# # plt.plot(x[spline_mask_1], conv_1[spline_mask_1], alpha=0.5, marker='o', ls='', ms=2)
# # plt.plot(x[spline_mask_2], conv_2[spline_mask_2], alpha=0.5, marker='o', ls='', ms=2)
# plt.plot(x, conv_1, alpha=0.5)
# plt.plot(x, conv_2, alpha=0.5)
# plt.plot(x, spl1(x))
# plt.plot(x, spl2(x))

# plt.show()
# plt.clf()

#%% Dall'Aglio Spline Method

# Divide spectrum into 16 pixel segments

def split_spec(len_spec, len_seg):
    
    segs_idxs = []
    
    n_segs = int(len_spec / len_seg)
    for n in range(1, n_segs+1):
        seg_start = (n-1)*len_seg
        seg_end = (n*len_seg) -1
        new_seg = np.linspace(seg_start, seg_end, len_seg)
        segs_idxs.append(new_seg.astype(int))

    return(segs_idxs)

def find_segs_cont(wave, flux, npix=25, s=0.5, k=3, noise_min=1125, noise_max=1175):
    
    spec_split = split_spec(len(wave), npix)

    # Fit continuous cubic spline to each segment
    
    spl_fits1 = []
    
    for i, segment in enumerate(spec_split):
        # get boundary conditions
        #bbox = [1, 1]
        spl1 = UnivariateSpline(wave[segment], flux[segment], s=s, k=k)
        # spl1 = CubicSpline(x[segment], conv_1[segment])
        
        spl_fits1.append(spl1)
        
    plt.figure(dpi=200)
    plt.plot(wave, flux, alpha=0.5)
    # plt.plot(x, conv_2, alpha=0.5)
    for i, segment in enumerate(spec_split):
        plt.plot(wave[segment], spl_fits1[i](x[segment]), color='blue')
    plt.ylim(-0.01, 2.1)
    plt.show()
    plt.clf()
    
    # Reject piexels in each segment which lie more than two sigma below the fit
    
    for i, segment in enumerate(spec_split): # refit one segment at a time
    
        seg_noise = np.std(flux[segment])
        spec_noise = get_noise(wave, conv_1, noise_min, noise_max)
        print('spec_noise = ', spec_noise)
    
        segment_ini = segment
        
        spl_flux = spl_fits1[i](wave[segment])
        seg_flux = flux[segment]
        seg_wave = wave[segment]
        sigma_flux = np.std(seg_flux)
        # print(sigma_flux)
        
        flux_dists = seg_flux - spl_flux
        flux_dists_sigma = flux_dists / sigma_flux
        print(flux_dists_sigma)
        
        # big_sig_mask = np.where(flux_dists_sigma <= -2.0)
        big_sig_mask = np.where(flux_dists <= -spec_noise)
        print('there are', len(big_sig_mask[0]), 'big sigs in segment', i)
        print(big_sig_mask[0])
        
        big_counter = len(big_sig_mask[0])
        while big_counter > 0:
            print('refitting segment', i)
            print(big_sig_mask[0])
            segment = np.delete(segment, big_sig_mask[0])
            
            seg_flux = flux[segment]
            seg_wave = wave[segment]
            
            spl_refit = UnivariateSpline(seg_wave, seg_flux, s=s, k=k)
            
            spl_fits1[i] = spl_refit
            
            spl_flux = spl_refit(x[segment])
            
            sigma_flux = np.std(seg_flux)
            
            flux_dists = seg_flux - spl_flux
            flux_dists_sigma = flux_dists / sigma_flux
            
            # big_sig_mask = np.where(flux_dists_sigma <= -2.0)
            big_sig_mask = np.where(flux_dists <= -spec_noise)
            big_counter = len(big_sig_mask[0])
            
    plt.figure(dpi=200)
    plt.plot(wave, flux, alpha=0.5)
    # plt.plot(x, conv_2, alpha=0.5)
    for i, segment in enumerate(spec_split):
        plt.plot(wave[segment], spl_fits1[i](x[segment]), color='blue')
    plt.ylim(-0.01, 2.1)
    plt.show()
    plt.clf()     
    
    
    spl_model_out = np.array([])
    for i, segment in enumerate(spec_split):
        spl_model_out = np.concatenate((spl_model_out, spl_fits1[i](x[segment])))

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
        con = ({'type': 'eq', 'fun': lambda c: x0 - anchor}, \
               {'type': 'eq', \
                'fun': lambda c: splev([x_end], (t, c, k), der=1)}, \
               {'type': 'eq', \
                'fun': lambda c: splev([x0], (t, c, k), der=2)})
        # con = ({'type': 'eq', 'fun': lambda c: x0 - anchor})
    else:
        print('not anchoring')
        con = ({'type': 'eq', \
               'fun': lambda c: splev([x_end], (t, c, k), der=1)}, \
               {'type': 'eq', \
                'fun': lambda c: splev([x0], (t, c, k), der=2)})
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))

def find_segs_cont_bound(wave, flux, npix=30, s=0.75, k=3, noise_min=1125, noise_max=1175):
    
    spec_split = split_spec(len(wave), npix)

    # Fit continuous cubic spline to each segment
    
    spl_fits1 = []
    
    for i, segment in enumerate(spec_split):
        
        # sp0 = UnivariateSpline(wave[segment], flux[segment], k=k, s=s)
        if i==0:
            anchor=None
        elif ((spl_fits1[i-1](x[spec_split[i-1]]))[-1] \
              > 1+ 1.5*get_noise(wave, conv_1, noise_min, noise_max)):
            anchor=1
        else:
            anchor=(spl_fits1[i-1](x[spec_split[i-1]]))[-1]
        spl1 = spline_neumann(wave[segment], flux[segment], k=k, s=s, anchor=anchor)
        # spl1 = UnivariateSpline(wave[segment], flux[segment], s=s, k=k)
        # spl1 = CubicSpline(x[segment], conv_1[segment])
        
        spl_fits1.append(spl1)
        
    plt.figure(dpi=200)
    plt.plot(wave, flux, alpha=0.5)
    # plt.plot(x, conv_2, alpha=0.5)
    for i, segment in enumerate(spec_split):
        plt.plot(wave[segment], spl_fits1[i](x[segment]), color='blue')
    plt.ylim(-0.01, 2.1)
    plt.show()
    plt.clf()
    
    # Reject piexels in each segment which lie more than two sigma below the fit
    
    for i, segment in enumerate(spec_split): # refit one segment at a time
        
        if i==0:
            anchor=None
            print('anchor set to none')
        elif ((spl_fits1[i-1](x[spec_split[i-1]]))[-1] \
              > 1+ 1.5*get_noise(wave, conv_1, noise_min, noise_max)):
            anchor=1
            print('anchor set to', anchor)
        else:
            anchor=(spl_fits1[i-1](x[spec_split[i-1]]))[-1]
            # print((spl_fits1[i-1](spec_split[i-1])))
            print('anchor set to', anchor)
    
        # seg_noise = np.std(flux[segment])
        spec_noise = get_noise(wave, conv_1, noise_min, noise_max)
        
    
        # segment_ini = segment
        
        spl_flux = spl_fits1[i](wave[segment])
        seg_flux = flux[segment]
        seg_wave = wave[segment]
        sigma_flux = np.std(seg_flux)
        # print(sigma_flux)
        
        flux_dists = seg_flux - spl_flux
        flux_dists_sigma = flux_dists / sigma_flux
        
        
        # big_sig_mask = np.where(flux_dists_sigma <= -2.0)
        big_sig_mask = np.where(flux_dists <= -spec_noise)
        
        
        big_counter = len(big_sig_mask[0])
        while big_counter > 0:
            
            segment = np.delete(segment, big_sig_mask[0])
            
            seg_flux = flux[segment]
            seg_wave = wave[segment]
            
            # sp0_refit = UnivariateSpline(wave[segment], flux[segment], k=k, s=s)
            # spl1 = spline_neumann(wave[segment], flux[segment], k=k, s=s)
            spl_refit = spline_neumann(seg_wave, seg_flux, s=s, k=k, anchor=anchor)
            
            spl_fits1[i] = spl_refit
            
            spl_flux = spl_refit(x[segment])
            
            sigma_flux = np.std(seg_flux)
            
            flux_dists = seg_flux - spl_flux
            flux_dists_sigma = flux_dists / sigma_flux
            
            big_sig_mask = np.where(flux_dists_sigma <= -2.0)
            # big_sig_mask = np.where(flux_dists <= -spec_noise)
            big_counter = len(big_sig_mask[0])
            
    plt.figure(dpi=200)
    plt.plot(wave, flux, alpha=0.5)
    # plt.plot(x, conv_2, alpha=0.5)
    for i, segment in enumerate(spec_split):
        plt.plot(wave[segment], spl_fits1[i](x[segment]), color='blue')
    plt.ylim(-0.01, 2.1)
    plt.show()
    plt.clf()     
    
    
    spl_model_out = np.array([])
    for i, segment in enumerate(spec_split):
        spl_model_out = np.concatenate((spl_model_out, spl_fits1[i](x[segment])))

    return(spl_model_out)

spl_model_1 = find_segs_cont_bound(x, conv_1)
spl_model_2 = find_segs_cont_bound(x, conv_2)