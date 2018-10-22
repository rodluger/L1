from tqdm import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl

def getData():
    data = np.load("../c9.npz")
    time = data['time']
    target_lcs = data['target_lcs']
    lcs = data['lcs']
    return time, target_lcs[2], lcs


class Detrender():

    def __init__(self, y, X, lam_idx, l1=True, mask=None, lam0=1e-6):

        # Tensorflow setup
        T = tf.float32
        np.random.seed(42)

        # Shapes
        nt = X.shape[0]
        nr = X.shape[1]
        nlam = len(lam_idx)
        if not hasattr(l1, "__len__"):
            l1 = [l1 for i in range(nlam)]

        self.lam_idx = lam_idx

        # Initial model guess: L2 with just a
        # tiny bit of regularization
        XTX = np.dot(np.transpose(X), X)
        XTX[np.diag_indices_from(XTX)] += lam0
        w0 = np.linalg.solve(XTX, np.dot(np.transpose(X), y))

        # New TF session
        self.session = tf.get_default_session()
        if self.session is None:
            self.session = tf.InteractiveSession()

        # Target flux, design matrix, and outlier mask
        self.y = tf.constant(y, dtype=T)
        self.X = tf.constant(X, dtype=T)
        if mask is None:
            mask = np.ones(nt, dtype=bool)
        self.mask = tf.constant(mask, dtype=tf.bool)

        # Weights
        self.w = tf.Variable(w0, dtype=T)

        # Regularization weights
        self.lambdas = tf.constant(np.ones(nlam), dtype=T)

        # Our linear model
        self.model = tf.squeeze(tf.matmul(self.X, self.w[:, None]))

        # Likelihood
        self.loss0 = tf.reduce_sum(tf.boolean_mask((self.y - self.model) ** 2,
                                   self.mask))

        # Penalty
        self.loss1 = 0
        for i in range(nlam):
            idx = lam_idx[i]
            if l1[i]:
                self.loss1 += self.lambdas[i] * tf.reduce_sum(tf.abs(self.w[idx]))
            else:
                self.loss1 += self.lambdas[i] * tf.reduce_sum(self.w[idx] ** 2)

        # Actual loss function
        self.loss = self.loss0 + self.loss1

        # Simple gradient descent optimizer
        self.learning_rate = tf.constant(1e-4, dtype=T)
        self.adam = tf.train.AdamOptimizer(self.learning_rate)
        self.opt_adam = self.adam.minimize(self.loss)

        # Woodbury identity + Zé's method to get weights
        '''
        L = tf.zeros(0, dtype=T)
        XLXT = tf.zeros([nt, nt], dtype=T)
        for i in range(nlam):
            idx = lam_idx[i]
            sz = idx.stop - idx.start
            if l1[i]:
                bk = tf.reduce_sum(tf.abs(self.w[idx]))
            else:
                bk = 1.
            XLXT += bk / self.lambdas[i] * tf.matmul(self.X[:, idx], self.X[:, idx],
                                         transpose_b=True)
            L = tf.concat((L, bk / self.lambdas[i] + tf.zeros(sz, dtype=T)), 0)
        M1 = L[:, None] * tf.transpose(self.X)
        M2 = tf.linalg.solve(XLXT + tf.eye(len(time)),
                             tf.reshape(self.y, [-1, 1]))
        '''

        XTX = tf.matmul(self.X, self.X, transpose_a=True)
        XTy = tf.matmul(self.X, self.y[:, None], transpose_a=True)
        L = tf.zeros(0, dtype=T)
        for i in range(nlam):
            idx = lam_idx[i]
            sz = idx.stop - idx.start
            if l1[i]:
                bk = tf.reduce_sum(tf.abs(self.w[idx]))
            else:
                bk = 1.
            L = tf.concat((L, self.lambdas[i] / bk + tf.zeros(sz, dtype=T)), 0)
        self.wk = tf.squeeze(tf.linalg.solve(XTX + tf.diag(L), XTy))

        self.opt_ze = tf.assign(self.w, self.wk)

        # Initialize
        self.initialize()

    def initialize(self):

        # Initialize the session
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)


def DetrendZe(detrender, lambdas=[1e1, 1e1, 1e-5], nsteps=50):
    """

    """
    detrender.initialize()
    feed_dict = {detrender.lambdas: lambdas}

    # Zé's method
    losses = []
    losses0 = []
    old_loss = detrender.loss.eval(feed_dict=feed_dict)
    old_w = detrender.w.eval()
    for j in range(nsteps):
        detrender.session.run(detrender.opt_ze, feed_dict=feed_dict)
        new_loss = detrender.loss.eval(feed_dict=feed_dict)
        '''
        if new_loss > old_loss:
            fd = dict(feed_dict)
            fd[detrender.wk] = old_w
            detrender.session.run(detrender.opt_ze, feed_dict=fd)
            break
        '''
        losses.append(new_loss)
        losses0.append(detrender.loss0.eval(feed_dict=feed_dict))
        old_loss = new_loss
        old_w = detrender.w.eval()

    # Show losses
    fig = pl.figure(figsize=(12, 4))
    pl.plot(losses, alpha=0.5);
    pl.plot(losses0, alpha=0.5);
    pl.yscale("log");
    pl.xlabel("Iteration");
    pl.ylabel("Loss");

    # Show weights
    fig, ax = pl.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(detrender.w.eval()[detrender.lam_idx[0]]);
    ax[1].plot(detrender.w.eval()[detrender.lam_idx[1]]);
    ax[2].plot(detrender.w.eval()[detrender.lam_idx[2]]);
    ax[0].set_ylabel("weight")
    for axis in ax:
        ymin, ymax = axis.get_ylim()
        ymax = max(np.abs(ymin), ymax)
        axis.set_ylim(-ymax, ymax)
        axis.set_xlabel("index")

    # Show model and data
    fig = pl.figure(figsize=(12, 4))
    pl.plot(time, 1 + detrender.y.eval());
    pl.plot(time, 1 + detrender.model.eval(feed_dict=feed_dict));
    pl.xlabel("time");
    pl.ylabel("flux");

    # Show de-trended
    fig = pl.figure(figsize=(12, 4))
    mod_lc1 = np.dot(detrender.X.eval()[:, detrender.lam_idx[0]], detrender.w.eval()[detrender.lam_idx[0]])
    mod_lc2 = np.dot(detrender.X.eval()[:, detrender.lam_idx[1]], detrender.w.eval()[detrender.lam_idx[1]])
    mod_pol = np.dot(detrender.X.eval()[:, detrender.lam_idx[2]], detrender.w.eval()[detrender.lam_idx[2]])
    pl.plot(time, 1 + detrender.y.eval() - (mod_lc1 + mod_lc2));
    pl.xlabel("time");
    pl.ylabel("flux");

    pl.show()

time, y, lcs = getData()
X = np.hstack([lcs.T,
               lcs.T ** 2,
               np.vander(np.linspace(0, 1, len(time)), N=5, increasing=True)])
lam_idx = [slice(0, len(lcs)),
           slice(len(lcs), 2 * len(lcs)),
           slice(2 * len(lcs), 2 * len(lcs) + 5)]
detrender = Detrender(y, X, lam_idx, [True, True, False])

DetrendZe(detrender)
