#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:03:25 2022

@author: user1
"""
import numpy as np
import matplotlib.pyplot as plt

fontsize = 20
figsize = (10,6)

# Configure parameters
plt.rcParams.update({'font.size': fontsize, 'figure.figsize': figsize})

# Default tick label size
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.linewidth'] = 2

def gauss(wave, *p):
    
    norm, mu, sigma, shift = p
    
    curve = shift+norm*np.exp(-(wave-mu)**2/(2.*sigma**2))
    
    return(curve)

wave = np.linspace(0, 10, 1000)

p0 = np.array((-1, 5, 0.5, 1))
p1 = np.array((-1, 3, 0.5, 1))

p2 = np.array((-1, 8, 0.5, 1))

abs0 = gauss(wave, *p0)
abs1 = gauss(wave, *p1)
abs2 = gauss(wave, *p2)

plt.figure(dpi=200)
plt.plot(wave,abs0, color='red', alpha=0.2, lw=3)
plt.plot(wave,abs1, color='blue', alpha=0.6, lw=3)
# plt.plot(wave,abs2, color='fuchsia', lw=3)

plt.xlabel('Wavelength')
plt.ylabel('Flux')

plt.show()
plt.clf()
