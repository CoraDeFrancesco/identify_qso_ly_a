#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:08:27 2022

@author: user1
"""
import numpy as np
import matplotlib.pyplot as plt
import sherpa.astro.models
from sherpa.models import Polynom1D
from sherpa.astro.models import Lorentz1D
from sherpa.data import Data1D
from sherpa.optmethods import NelderMead
from sherpa.stats import LeastSq
from sherpa.data import DataSimulFit
from sherpa.models import SimulFitModel
from sherpa.fit import Fit

##############################################################################
# Custom Model
##############################################################################

from sherpa.models import model

__all__ = ('LyNorm', )


def _LyNorm(pars, x):
    """Use Bosman eivenvectors to normalize Ly-a forest region.

    Parameters
    ----------
    pars: sequence of 4 numbers
        The order is amplitude, center, width, and slope.
        These numbers are assumed to be valid (e.g. width
        is 0 or greater).
    x: sequence of numbers
        The grid on which to evaluate the model. It is expected
        to be a floating-point type.

    Returns
    -------
    y: sequence of numbers
        The model evaluated on the input grid.

    Notes
    -----
    This is based on the interface described at
    https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Trapezoid1D.html
    but implemented without looking at the code, so any errors
    are not due to AstroPy.
    """

    (amplitude, center, width, slope) = pars

    # There are five segments:
    #    xlo = center - width/2
    #    xhi = center + width/2
    #    x0  = xlo - amplitude/slope
    #    x1  = xhi + amplitude/slope
    #
    #    flat   xlo <= x < xhi
    #    slope  x0 <= x < xlo
    #           xhi <= x < x1
    #    zero   x < x0
    #           x >= x1
    #
    hwidth = width / 2.0
    dx = amplitude / slope
    xlo = center - hwidth
    xhi = center + hwidth
    x0 = xlo - dx
    x1 = xhi + dx

    out = np.zeros(x.size)
    out[(x >= xlo) & (x < xhi)] = amplitude

    idx = np.where((x >= x0) & (x < xlo))
    out[idx] = slope * x[idx] - slope * x0

    idx = np.where((x >= xhi) & (x < x1))
    out[idx] = - slope * x[idx] + slope * x1

    return out


class Trap1D(model.RegriddableModel1D):
    """A one-dimensional trapezoid.

    The model parameters are:

    ampl
        The amplitude of the central (flat) segment (zero or greater).
    center
        The center of the central segment.
    width
        The width of the central segment (zero or greater).
    slope
        The gradient of the slopes (zero or greater).

    """

    def __init__(self, name='trap1d'):
        self.ampl = model.Parameter(name, 'ampl', 1, min=0, hard_min=0)
        self.center = model.Parameter(name, 'center', 1)
        self.width = model.Parameter(name, 'width', 1, min=0, hard_min=0)
        self.slope = model.Parameter(name, 'slope', 1, min=0, hard_min=0)

        model.RegriddableModel1D.__init__(self, name,
                                          (self.ampl, self.center, self.width,
                                           self.slope))

    def calc(self, pars, x, *args, **kwargs):
        """Evaluate the model"""

        # If given an integrated data set, use the center of the bin
        if len(args) == 1:
            x = (x + args[0]) / 2

        return _trap1d(pars, x)
    
##############################################################################

tpoly = Polynom1D()
tlor = Lorentz1D()
tpoly.c0 = 50
tpoly.c1 = 1e-2
tlor.pos = 4400
tlor.fwhm = 200
tlor.ampl = 1e4
x1 = np.linspace(4200, 4600, 21)
y1 = tlor(x1) + tpoly(x1) + np.random.normal(scale=5, size=x1.size)
x2 = np.linspace(4100, 4900, 11)
y2 = tpoly(x2) + np.random.normal(scale=5, size=x2.size)
print("x1 size {}  x2 size {}".format(x1.size, x2.size))

plt.plot(x1, y1)
plt.plot(x2, y2)

d1 = Data1D('a', x1, y1)
d2 = Data1D('b', x2, y2)
fpoly, flor = Polynom1D(), Lorentz1D()
fpoly.c1.thaw()
flor.pos = 4500

flor.ampl = y1.sum() / flor(x1).sum()

stat, opt = LeastSq(), NelderMead()

f1 = Fit(d1, fpoly + flor, stat, opt)
f2 = Fit(d2, fpoly, stat, opt)

sdata = DataSimulFit('all', (d1, d2))
smodel = SimulFitModel('all', (fpoly + flor, fpoly))
sfit = Fit(sdata, smodel, stat, opt)

res = sfit.fit()
print(res)



