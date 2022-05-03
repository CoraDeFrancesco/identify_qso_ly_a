#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:56:39 2022

@author: user1
"""
import numpy as np
import matplotlib.pyplot as plt

def gauss(wave, *p):
    
    norm, mu, sigma, shift = p
    
    curve = shift+norm*np.exp(-(wave-mu)**2/(2.*sigma**2))
    
    return(curve)


bnorm = -0.75
bmu = 50
bsigma = 5
bshift = 1

bnorm2 = -0.4
bmu2 = 49.5
bsigma2 = 5
bshift2 = 1


lnorm = -0.4
lmu = 55
lsigma = 0.4
lshift = 1

x = np.linspace(0, 100, 1000)

bline = gauss(x, bnorm, bmu, bsigma, bshift)
bline2 = gauss(x, bnorm2, bmu2, bsigma2, bshift2)
lline = gauss(x, lnorm, lmu, lsigma, lshift)

ly_min_mask = np.where(lline == min(lline))

conv_1 = bline*lline
conv_2 = bline2*lline

plt.figure(dpi=200)

plt.plot(x, bline, color='red', alpha=0.5)
plt.plot(x, bline2, color='purple', alpha=0.5)

plt.plot(x, lline, label='Ly-a', lw=3, color='green')
plt.plot(x,conv_1, label='Observed 1', color='red', lw=3)
plt.plot(x,conv_2, label='Observed 2', color='purple', lw=3)

# plt.plot(x[ly_min_mask],conv_1[ly_min_mask], marker='*', color='red')
# plt.plot(x[ly_min_mask],conv_2[ly_min_mask], marker='*', color='purple')


plt.legend()

plt.ylim(-0.01, 1.25)
plt.xlim(20, 80)

plt.show()
plt.clf()

print('Observed 1 broad min:', -bnorm)
print('            Ly-a min:', conv_1[ly_min_mask])   
print('Observed 2 broad min:', -bnorm2)
print('            Ly-a min:', conv_2[ly_min_mask])   

print('Ratio 1:', (-bnorm / conv_1[ly_min_mask]))
print('Ratio 2:', (-bnorm2 / conv_2[ly_min_mask]))

print('Ly ratio = ', ((-bnorm / conv_1[ly_min_mask])/(-bnorm / conv_2[ly_min_mask])))
