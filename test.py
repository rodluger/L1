import numpy as np
np.random.seed(42)
import matplotlib.pyplot as pl
from astropy.io import fits as pyfits
import glob
from scipy.signal import savgol_filter
from scipy.ndimage.measurements import label
from scipy.ndimage import zoom
from tqdm import tqdm
import random
import os
import everest
import celeriteflow as cf
import keras
bad_bits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17]


def GetFiles(clobber=False, kp_min=11, kp_max=16, snr=10):
    """
    Get a list of all files with targets in the desired Kp range
    and some extra cuts.

    """
    # Load saved?
    if (os.path.exists("data/c6/files.txt") and not clobber):
        with open("data/c6/files.txt", "r") as f:
            return [line[:-1] for line in f.readlines()]
    all_files = glob.glob("data/c6/*/*/*lpd-targ.fits.gz")

    # Make cuts
    files = []
    for file in all_files:

        # Magnitude cut
        try:
            kp = float(pyfits.getheader(file, 0)["kepmag"])
            if (kp < kp_min) or (kp > kp_max):
                continue
        except TypeError:
            continue

        # Grab the data
        with pyfits.open(file) as f:
            qdata = f[1].data

        # Identify all sources from the median TPF image
        img = np.nanmedian(qdata.field('FLUX'), axis=0)
        mu = np.nanmedian(img)
        sigma = np.sqrt(np.nanmedian((img - mu) ** 2))

        # Cut: require a single source
        m = (img - mu) > snr * sigma
        labels, nstar = label(m)
        if (nstar != 1):
            continue

        # Cut: require the aperture to not touch the edges
        # of the postage stamp
        rows, cols = m.shape
        touches = np.count_nonzero(m[rows - 1, :]) + \
                  np.count_nonzero(m[0, :]) + \
                  np.count_nonzero(m[:, cols - 1]) + \
                  np.count_nonzero(m[:, 0])
        if touches != 0:
            continue

        # We're good
        files.append(file)

    # Save and return
    with open("data/c6/files.txt", "w") as f:
        for file in files:
            print(file, file=f)
    return files


def Smooth(x, window_len=100, window='hanning'):
    '''
    Smooth data by convolving on a given timescale.

    :param ndarray x: The data array
    :param int window_len: The size of the smoothing window. Default `100`
    :param str window: The window type. Default `hanning`

    '''
    if window_len == 0:
        return np.zeros_like(x)
    s = np.r_[2 * x[0] - x[window_len - 1::-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]


def Interpolate(time, mask, y):
    '''
    Masks certain elements in the array `y` and linearly
    interpolates over them, returning an array `y'` of the
    same length.

    :param array_like time: The time array
    :param array_like mask: The indices to be interpolated over
    :param array_like y: The dependent array

    '''
    # Ensure `y` doesn't get modified in place
    yy = np.array(y)
    t_ = np.delete(time, mask)
    y_ = np.delete(y, mask, axis=0)
    if len(yy.shape) == 1:
        yy[mask] = np.interp(time[mask], t_, y_)
    elif len(yy.shape) == 2:
        for n in range(yy.shape[1]):
            yy[mask, n] = np.interp(time[mask], t_, y_[:, n])
    else:
        raise Exception("Array ``y`` must be either 1- or 2-d.")
    return yy


def Chunks(l, n, all=False):
    '''
    Returns a generator of consecutive `n`-sized chunks of list `l`.
    If `all` is `True`, returns **all** `n`-sized chunks in `l`
    by iterating over the starting point.

    '''
    if all:
        jarr = range(0, n - 1)
    else:
        jarr = [0]

    for j in jarr:
        for i in range(j, len(l), n):
            if i + 2 * n <= len(l):
                yield l[i:i + n]
            else:
                if not all:
                    yield l[i:]
                break


def Scatter(y, win=13, remove_outliers=True):
    '''
    Return the scatter in ppm based on the median running standard deviation
    for a window size of :py:obj:`win` = 13 cadences (for K2, this
    is ~6.5 hours, as in VJ14).

    :param ndarray y: The array whose CDPP is to be computed
    :param int win: The window size in cadences. Default `13`
    :param bool remove_outliers: Clip outliers at 5 sigma before computing \
           the CDPP? Default `False`

    '''
    if remove_outliers:
        # Remove 5-sigma outliers from data
        # smoothed on a 1 day timescale
        if len(y) >= 50:
            ys = y - Smooth(y, 50)
        else:
            ys = y
        M = np.nanmedian(ys)
        MAD = 1.4826 * np.nanmedian(np.abs(ys - M))
        out = []
        for i, _ in enumerate(y):
            if (ys[i] > M + 5 * MAD) or (ys[i] < M - 5 * MAD):
                out.append(i)
        out = np.array(out, dtype=int)
        y = np.delete(y, out)
    if len(y):
        return 1.e6 * np.nanmedian([np.std(yi) / np.sqrt(win)
                                    for yi in Chunks(y, win, all=True)])
    else:
        return np.nan


def SavGol(y, win=49):
    '''
    Subtracts a second order Savitsky-Golay filter with window size `win`
    and returns the result. This acts as a high pass filter.

    '''
    if len(y) >= win:
        return y - savgol_filter(y, win, 2) + np.nanmedian(y)
    else:
        return y


def PadWithZeros(vector, pad_width, iaxis, kwargs):
    '''

    '''
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def GetRegressors(files, clobber=False, snr=10, plot=True):
    """Get the regressors."""
    if os.path.exists("data/c6/regressors.npz") and not clobber:
        data = np.load("data/c6/regressors.npz")
        return data["fluxes"], data["pixels"], data["npixels"]

    if plot:
        fig, ax = pl.subplots(10, 10, figsize=(16, 8))
        ax = ax.flatten()
        nax = 0
        nfig = 0

    # Get the flux arrays
    fluxes = []
    fluxes_err = []
    pixels = []
    pixels_err = []
    for i, reg in tqdm(enumerate(files), total=len(files)):
        with pyfits.open(reg) as f:

            # Grab the data
            qdata = f[1].data

            # Get a small aperture
            img = np.nanmedian(qdata.field('FLUX'), axis=0)
            mu = np.nanmedian(img)
            sigma = np.sqrt(np.nanmedian((img - mu) ** 2))
            m = (img - mu) > snr * sigma
            labels, nstar = label(m)

            # Enlarge it
            aperture = np.array(m)
            for i in range(2):
                aperture = aperture | np.vstack((m[1:, :], np.zeros(m.shape[1], dtype=bool)))
                aperture = aperture | np.hstack((m[:, 1:], np.zeros(m.shape[0], dtype=bool).reshape(-1, 1)))
                aperture = aperture | np.vstack((np.zeros(m.shape[1], dtype=bool), m))[:-1, :]
                aperture = aperture | np.hstack((np.zeros(m.shape[0], dtype=bool).reshape(-1, 1), m))[:, :-1]
                m = aperture

            # Plot the target?
            if plot:
                ax[nax].imshow(img, origin='lower', alpha=1)
                contour = np.zeros_like(aperture)
                contour[np.where(aperture)] = 1
                contour = np.lib.pad(contour, 1, PadWithZeros)
                highres = zoom(contour, 100, order=0, mode='nearest')
                extent = np.array([-1, m.shape[1], -1, m.shape[0]])
                contour = ax[nax].contour(highres, levels=[0.5], extent=extent,
                                          origin='lower', colors='r',
                                          linewidths=1)

                # Save the figure if we've filled all axes
                nax += 1
                if (nax == len(ax)) or (i == len(files) - 1):
                    for axis in ax:
                        axis.axis('off')
                    fig.savefig("data/c6/regressors%02d.pdf" % nfig, bbox_inches="tight")
                    pl.close()
                    fig, ax = pl.subplots(10, 10, figsize=(16, 8))
                    ax = ax.flatten()
                    nax = 0
                    nfig += 1

            # Get the arrays
            cadn = np.array(qdata.field('CADENCENO'), dtype='int32')
            time = np.array(qdata.field('TIME'), dtype='float64')
            fpix3D = np.array(qdata.field('FLUX'), dtype='float64')
            fpix3D_err = np.array(qdata.field('FLUX_ERR'), dtype='float64')
            qual = np.array(qdata.field('QUALITY'), dtype=int)

            # Get rid of NaNs in the time array by interpolating
            naninds = np.where(np.isnan(time))
            time = Interpolate(np.arange(0, len(time)), naninds, time)

            # Flatten the pixel array
            aperture[np.isnan(fpix3D[0])] = 0
            ap = np.where(aperture & 1)
            fpix = np.array([f[ap] for f in fpix3D], dtype='float64')
            fpix_err = np.array([p[ap] for p in fpix3D_err], dtype='float64')
            flux = np.sum(fpix, axis=1)
            flux_err = np.sqrt(np.sum(fpix_err ** 2, axis=1))

            # Interpolate over NaNs in the flux
            nanmask = np.where(np.isnan(flux) | (flux == 0))[0]
            fpix = Interpolate(time, nanmask, fpix)
            fpix_err = Interpolate(time, nanmask, fpix_err)
            flux = Interpolate(time, nanmask, flux)
            flux_err = Interpolate(time, nanmask, flux_err)

            # Interpolate over quality flags
            badmask = []
            for b in bad_bits:
                badmask += list(np.where(qual & 2 ** (b - 1))[0])
            fpix = Interpolate(time, badmask, fpix)
            fpix_err = Interpolate(time, badmask, fpix_err)
            flux = Interpolate(time, badmask, flux)
            flux_err = Interpolate(time, badmask, flux_err)

            # Interpolate over >10 sigma outliers
            f = SavGol(flux)
            med = np.nanmedian(f)
            MAD = 1.4826 * np.nanmedian(np.abs(f - med))
            badmask = np.where((f > med + 10. * MAD) | (f < med - 10. * MAD))[0]
            fpix = Interpolate(time, badmask, fpix)
            fpix_err = Interpolate(time, badmask, fpix_err)
            flux = Interpolate(time, badmask, flux)
            flux_err = Interpolate(time, badmask, flux_err)

            # Normalize everything to unit median flux
            norm = np.nanmedian(flux)
            flux /= norm
            flux_err /= norm
            fpix /= norm
            fpix_err /= norm

            # Save to list of regressors
            fluxes.append(flux)
            fluxes_err.append(flux_err)
            pixels.append(fpix)
            pixels_err.append(fpix_err)

    # Flatten things
    fluxes = np.array(fluxes)
    fluxes_err = np.array(fluxes_err)
    npixels = [p.shape[1] for p in pixels]
    pixels = np.hstack(pixels).transpose()
    pixels_err = np.hstack(pixels_err).transpose()

    # Save to disk and return
    np.savez("data/c6/regressors.npz", fluxes=fluxes, fluxes_err=fluxes_err,
             pixels=pixels, pixels_err=pixels_err, npixels=npixels)
    return fluxes, pixels, npixels


def GetTarget(epicid, snr=10, plot=True):
    """Get the target we're going to de-trend."""
    file = os.path.join('data', 'c6', ('%09d' % epicid)[:4] + '00000',
             ('%09d' % epicid)[4:6] + '000', "ktwo%d-c06_lpd-targ.fits.gz" % epicid)
    with pyfits.open(file) as f:

        # Grab the data
        qdata = f[1].data
        epicid = int(f[0].header["KEPLERID"])

        # Get a small aperture
        img_med = np.nanmedian(qdata.field('FLUX'), axis=0)
        mu = np.nanmedian(img_med)
        sigma = np.sqrt(np.nanmedian((img_med - mu) ** 2))
        m = (img_med - mu) > snr * sigma
        labels, nstar = label(m)

        # Enlarge it
        aperture = np.array(m)
        for i in range(2):
            aperture = aperture | np.vstack((m[1:, :], np.zeros(m.shape[1], dtype=bool)))
            aperture = aperture | np.hstack((m[:, 1:], np.zeros(m.shape[0], dtype=bool).reshape(-1, 1)))
            aperture = aperture | np.vstack((np.zeros(m.shape[1], dtype=bool), m))[:-1, :]
            aperture = aperture | np.hstack((np.zeros(m.shape[0], dtype=bool).reshape(-1, 1), m))[:, :-1]
            m = aperture

        # Get the arrays
        cadn = np.array(qdata.field('CADENCENO'), dtype='int32')
        time = np.array(qdata.field('TIME'), dtype='float64')
        fpix3D = np.array(qdata.field('FLUX'), dtype='float64')
        fpix3D_err = np.array(qdata.field('FLUX_ERR'), dtype='float64')
        qual = np.array(qdata.field('QUALITY'), dtype=int)

        # Get rid of NaNs in the time array by interpolating
        naninds = np.where(np.isnan(time))
        time = Interpolate(np.arange(0, len(time)), naninds, time)

        # Flatten the pixel array
        aperture[np.isnan(fpix3D[0])] = 0
        ap = np.where(aperture & 1)
        fpix = np.array([f[ap] for f in fpix3D], dtype='float64')
        fpix_err = np.array([p[ap] for p in fpix3D_err], dtype='float64')
        flux = np.sum(fpix, axis=1)
        flux_err = np.sqrt(np.sum(fpix_err ** 2, axis=1))

        # Interpolate over NaNs in the flux
        nanmask = np.where(np.isnan(flux) | (flux == 0))[0]
        fpix = Interpolate(time, nanmask, fpix)
        fpix_err = Interpolate(time, nanmask, fpix_err)
        flux = Interpolate(time, nanmask, flux)
        flux_err = Interpolate(time, nanmask, flux_err)

        # Interpolate over quality flags
        badmask = []
        for b in bad_bits:
            badmask += list(np.where(qual & 2 ** (b - 1))[0])
        fpix = Interpolate(time, badmask, fpix)
        fpix_err = Interpolate(time, badmask, fpix_err)
        flux = Interpolate(time, badmask, flux)
        flux_err = Interpolate(time, badmask, flux_err)

        # Interpolate over >10 sigma outliers
        f = SavGol(flux)
        med = np.nanmedian(f)
        MAD = 1.4826 * np.nanmedian(np.abs(f - med))
        badmask = np.where((f > med + 10. * MAD) | (f < med - 10. * MAD))[0]
        fpix = Interpolate(time, badmask, fpix)
        fpix_err = Interpolate(time, badmask, fpix_err)
        flux = Interpolate(time, badmask, flux)
        flux_err = Interpolate(time, badmask, flux_err)

        # Normalize everything to unit median flux
        norm = np.nanmedian(flux)
        flux /= norm
        flux_err /= norm
        fpix /= norm
        fpix_err /= norm

        # Plot the target?
        if plot:

            # Setup
            fig = pl.figure(figsize=(15, 6))
            ax_ps = [pl.subplot2grid((6, 5), (0, 0), rowspan=2, colspan=1),
                     pl.subplot2grid((6, 5), (2, 0), rowspan=2, colspan=1),
                     pl.subplot2grid((6, 5), (4, 0), rowspan=2, colspan=1)]
            ax_lc = [pl.subplot2grid((6, 5), (0, 1), rowspan=3, colspan=4),
                     pl.subplot2grid((6, 5), (3, 1), rowspan=3, colspan=4)]
            ax_lc[0].set_xticklabels([])
            ax_lc[1].set_xlabel("Time", fontsize=14)

            # Plot the postage stamp
            fsap = np.nansum(qdata.field('FLUX'), axis=(1, 2))
            imed = np.nanargmin(np.abs(fsap - np.nanmedian(fsap)))
            for ax, img, title in zip(ax_ps,
                                      [qdata.field('FLUX')[imed],
                                       img_med,
                                       np.log10(np.clip(img_med, 0.1, None))],
                                      ["med", "lin", "log"]):
                ax.axis("off")
                ax.imshow(img, origin='lower', alpha=1)
                ax.set_title(title, y=0.95, fontsize=8)
                contour = np.zeros_like(aperture)
                contour[np.where(aperture)] = 1
                contour = np.lib.pad(contour, 1, PadWithZeros)
                highres = zoom(contour, 100, order=0, mode='nearest')
                extent = np.array([-1, m.shape[1], -1, m.shape[0]])
                contour = ax.contour(highres, levels=[0.5], extent=extent,
                                     origin='lower', colors='r',
                                     linewidths=1)

            # Plot the raw flux
            ax_lc[0].plot(time, flux, 'k.', alpha=0.3, ms=2,
                          label="%.3f ppm" % Scatter(flux))
            ax_lc[0].legend(loc="upper right")

            # Download and plot the EVEREST flux
            star = everest.Everest(epicid, campaign=6)
            ev_time = star.time
            ev_flux = star.flux /  np.nanmedian(star.flux)
            ax_lc[1].plot(ev_time, ev_flux, 'k.', alpha=0.3, ms=2,
                          label="%.3f ppm" % Scatter(ev_flux))
            ax_lc[1].legend(loc="upper right")
            ax_lc[1].set_ylim(*ax_lc[0].get_ylim())
            fig.savefig("data/c6/%d.pdf" % epicid, bbox_inches="tight")
            pl.close()

        # Return
        return time, flux, flux_err


def GetTestData(clobber=False):
    """Get test data."""
    if os.path.exists("data/c6/test_data.npz") and not clobber:
        data = np.load("data/c6/test_data.npz")
        reg_fluxes = data['reg_fluxes']
        reg_pixels = data['reg_pixels']
        reg_npix = data['reg_npix']
        time = data['time']
        flux = data['flux']
        flux_err = data['flux_err']
    else:
        files = GetFiles()
        reg_fluxes, reg_pixels, reg_npix = GetRegressors(files, plot=False)
        time, flux, flux_err = GetTarget(212394689)
        np.savez("data/c6/test_data.npz", reg_fluxes=reg_fluxes, reg_pixels=reg_pixels,
                 reg_npix=reg_npix, flux=flux, flux_err=flux_err, time=time)
    return time, reg_fluxes, reg_pixels, reg_npix, flux, flux_err


# Grab the data
time, reg_fluxes, reg_pixels, reg_npix, flux, flux_err = GetTestData()
ntime = flux.shape[0]
nreg = reg_fluxes.shape[0]

# Set up tensorflow
import tensorflow as tf
dtype = tf.float64

# Tweakable stuff
logjitter0 = np.log(0.01)  # Initial log jitter
l0 = 1e-2  # Initial L2 regularization variance
learning_rate = 1e-2  # Initial Adam learning rate
niter = 500  # Number of iterations
P0 = 10.0  # Period guess
H = 2  # Dimension of the hidden layer
def activate(t):
    return tf.clip_by_value(t, 0, np.inf)

# This is our design matrix
X = tf.constant(reg_fluxes.T - 1.0, dtype=dtype)

# This is our data we want to fit
y = tf.constant(flux - 1.0, dtype=dtype)

# This is the data uncertainty
y_err = tf.constant(flux_err, dtype=dtype)

# Compute weakly regularized max like solution as a guess
logjitter = tf.Variable(logjitter0, dtype=dtype)
L = tf.diag(tf.ones(nreg, dtype=dtype) * l0 ** 2)
LXT = tf.matmul(L, X, transpose_b=True)
S = tf.matmul(X, tf.matmul(L, X, transpose_b=True))
S += tf.diag(y_err**2 + tf.exp(2*logjitter))
Sinvy = tf.linalg.solve(S, y[:, None])
w0 = tf.matmul(LXT, Sinvy)

# Initial model
w1 = tf.Variable(np.zeros((nreg, H)), dtype=dtype)
w2 = tf.Variable(np.zeros((H, 1)), dtype=dtype)
model = tf.squeeze(tf.matmul(activate(tf.matmul(X, w1)), w2))
l = tf.constant(l0, dtype=dtype)

# Celerite GP
t = tf.constant(time, dtype=dtype)
diag = y_err ** 2 + tf.exp(2*logjitter)
resid = y - model
log_S0 = tf.Variable(np.log(np.var(flux)), dtype=dtype)
log_w0 = tf.Variable(np.log(2 * np.pi / P0), dtype=dtype)
log_Q = tf.Variable(0.0, dtype=dtype)
kernel = cf.terms.SHOTerm(log_S0=log_S0,
                          log_w0=log_w0,
                          log_Q=log_Q)
gp = cf.GaussianProcess(kernel, t, resid, diag)
loglike = gp.log_likelihood

# Losses
loss0 = -2 * loglike
loss1 = (1 / l) * tf.reduce_sum(tf.abs(w1))
loss2 = (1 / l) * tf.reduce_sum(tf.abs(w2))
loss = loss0 + loss1 + loss2
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Init session
session = tf.get_default_session()
if session is None:
    session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

w1_0 = 1e-6 * np.random.randn(nreg, H)
w1_0[:, 0] = w0.eval()[:, 0]
w2_0 = 1e-6 * np.random.randn(H, 1)
w2_0[0, 0] = 1

session.run(tf.assign(w1, w1_0))
session.run(tf.assign(w2, w2_0))
model0 = model.eval()

# Iterate!
losses = np.zeros(niter)
best_loss = np.inf
for i in tqdm(range(niter)):
    session.run(opt)
    losses[i] = loss.eval()
    if losses[i] < best_loss:
        best_loss = losses[i]
        best_w1 = w1.eval()
        best_w2 = w2.eval()
        best_logjitter = logjitter.eval()
        best_log_S0 = log_S0.eval()
        best_log_w0 = log_w0.eval()
        best_log_Q = log_Q.eval()
session.run(tf.assign(w1, best_w1))
session.run(tf.assign(w2, best_w2))
session.run(tf.assign(logjitter, best_logjitter))
session.run(tf.assign(log_S0, best_log_S0))
session.run(tf.assign(log_w0, best_log_w0))
session.run(tf.assign(log_Q, best_log_Q))

#
print("GP stuff:", best_logjitter, best_log_S0, best_log_w0, best_log_Q)

# Plot learning rate
fig, ax = pl.subplots(1)
ax.plot(range(niter), losses)

# Plot weights
fig, ax = pl.subplots(1)
ax.imshow(np.log10(np.abs(best_w1)), aspect='auto')

# Plot weights
fig, ax = pl.subplots(1)
ax.imshow(np.log10(np.abs(best_w2)), aspect='auto')

# Plot initial model
fig, ax = pl.subplots(2)
ax[0].set_title("Initial model")
ax[0].plot(time, flux, 'k.', ms=2, alpha=0.3)
ax[0].plot(time, 1 + model0, 'r-', lw=0.5)
ax[1].plot(time, flux - model0, 'k.', ms=2, alpha=0.3)

# Plot final model
fig, ax = pl.subplots(2)
ax[0].set_title("Final model")
ax[0].plot(time, flux, 'k.', ms=2, alpha=0.3)
ax[0].plot(time, 1 + model.eval(), 'r-', lw=0.5)
ax[1].plot(time, flux - model.eval(), 'k.', ms=2, alpha=0.3)

pl.show()
