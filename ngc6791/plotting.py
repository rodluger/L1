from astropy.io import fits as pyfits
import numpy as np
import matplotlib.pyplot as pl
import glob
import os
from tqdm import tqdm
from scipy.ndimage import zoom
from .apertures import EBaperture

__all__ = ["PlotImage"]

def PadWithZeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def PlotAperture(ax, aperture):
    rows, cols = aperture.shape
    contour = np.zeros((rows, cols))
    contour[np.where(aperture)] = 1
    contour = np.lib.pad(contour, 1, PadWithZeros)
    highres = zoom(contour, 20, order=0, mode='nearest')
    extent = np.array([-1, rows, -1, cols])
    ax.contour(highres, levels=[0.5], extent=extent,
               origin='lower', colors='r', linewidths=2)


def PlotImage(quarter):
    # Load the data
    ap = EBaperture(quarter)
    rows, cols = ap.shape
    file = glob.glob(os.path.join('data', 'ngc6791',
                                  'Q%d' % quarter, '*.fits'))[0]
    with pyfits.open(file) as f:
        flux = f[0].data

    # Plot the image + aperture
    fig = pl.figure()
    pl.imshow(flux, vmax=1e3)
    PlotAperture(pl.gca(), ap)
    pl.show()


if __name__ == '__main__':
    PlotImage(1)
