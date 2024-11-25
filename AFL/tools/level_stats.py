"""
Functions to compute level statistics using spectral unfolding.
"""

import numpy as np
import AFL.tools.floquet as floquet
from scipy.interpolate import make_smoothing_spline
from KDEpy import FFTKDE

# floquet handling, as well as possible mirroring for a bounded spectrum?
# Should the integral be done via quadrature (use TreeKDE?) or via sampling?
# Perhaps we should unfold via quadrature (more exact), but have a function to get the CSF quickly everywhere via sampling.
# Worry about FFTKDE binning, since that issue is what we are trying to fix in the first place.
def unfold_KDE(spectrum, return_KDE=False):
    """
    Given a spectrum, unfolds the spectrum so that the mean level spacing is unity.
    This function unfolds by estimating the smoothed spectral density via kernel density estimation.
    By default, we use a Gaussian kernel with improved Sheather and Jones for bandwidth estimation.
    """
    return None

# TODO: improve floquet fitting by repeating data?
def unfold_spline(spectrum, return_spline=False, lam=None):
    """
    Given a spectrum, unfolds the spectrum so that the mean level spacing is unity.
    This function unfolds by fitting a smoothing cubic spline to the cumulative spectral function (CSF).
    Returns the unfolded spectrum, and optionally the spline fit to the CSF.
    
    See scipy.interpolate.make_smoothing_spline for method details, including the 'lam' keyword.
    For the sample points, we pick the bottom and top of each step in the CSF.
    """
    sorted = np.sort(spectrum)
    unique_en, counts = np.unique(sorted, return_counts=True)
    cum_count = np.cumsum(counts)
    # At each energy, include the bottom and top of the step in the CSF
    x = np.tile(unique_en, 2)
    y = np.append(cum_count, cum_count - counts)
    spline = make_smoothing_spline(x, y, lam=lam)
    unfolded = spline(sorted)
    if return_spline:
        return unfolded, spline
    return unfolded
