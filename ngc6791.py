from astropy.io import fits as pyfits
import numpy as np
import matplotlib.pyplot as pl
import glob
import os
from tqdm import tqdm
from scipy.ndimage import zoom


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

def Plot(aperture, quarter='Q17'):
    # Load the data
    rows, cols = aperture.shape
    files = glob.glob(os.path.join('data', quarter, '*.fits'))
    time = np.empty(len(files))
    flux = np.empty((len(files), rows, cols))
    for i, file in tqdm(enumerate(files), total=len(files)):
        with pyfits.open(file) as f:
            time[i] = f[0].header['MIDTIME']
            flux[i] = f[0].data
    idx = np.argsort(time)
    time = time[idx]
    flux = flux[idx]

    # Plot the scene
    fig = pl.figure()
    pl.imshow(flux[0], vmax=1e3)
    PlotAperture(pl.gca(), aperture)

    # Plot our target flux
    fig = pl.figure()
    target_flux = flux[:, aperture]
    pl.plot(time, np.sum(target_flux, axis=(1)))

    # Show
    pl.show()


# Plot Q17
aperture = np.zeros((200, 200), dtype=bool)
for col in np.arange(164, 169):
    for row in np.arange(174, 178):
        aperture[row, col] = True
Plot(aperture, 'Q17')
