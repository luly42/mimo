import numpy as np
import h5py
from sklearn.neighbors import KDTree
from collections import namedtuple
from scipy.stats import norm as normal


def mid(x):
    return (x[1:] + x[:-1]) / 2.


def twod_interp(x, y, x0, x1, y0, y1):
    w0 = (x1 - x0) * (y1 - y0)
    w00 = (x1 - x) * (y1 - y) / w0
    w01 = (x1 - x) * (y - y0) / w0
    w10 = (x - x0) * (y1 - y) / w0
    w11 = (x - x0) * (y - y0) / w0

    return w00, w01, w10, w11


def pow_rvs(n, alpha, ymin, ymax):
    if abs(alpha + 1) >= 1e-5:
        xmax, xmin = ymax**(1 + alpha), ymin**(1 + alpha)
        x = np.random.rand(n) * (xmax - xmin) + xmin
        y = x**(1. / (1 + alpha))
    else:
        xmax, xmin = np.log(ymax), np.log(ymin)
        x = np.random.rand(n) * (xmax - xmin) + xmin
        y = np.exp(x)
    return y


def pow_pdf(y, alpha, ymin, ymax):
    if abs(alpha + 1) >= 1e-5:
        return (1. + alpha) / (ymax**(alpha + 1) - ymin**(alpha + 1)) * y**alpha
    else:
        return 1 / (np.log(ymax) - np.log(ymin)) * y**alpha


def pow_cdf(y, alpha, ymin, ymax):
    # if alpha != -1:
    if abs(alpha + 1) >= 1e-5:
        ymin, ymax = ymin ** (alpha + 1), ymax ** (alpha + 1)
        y = y**(alpha + 1)
    else:
        ymin, ymax = np.log(ymin), np.log(ymax)
        y = np.log(y)
    return (y - ymin) / (ymax - ymin)


def pow_med(alpha, ymin, ymax):
    # if alpha != -1:
    if abs(alpha + 1) >= 1e-5:
        ymin, ymax = ymin**(alpha + 1), ymax**(alpha + 1)
        return (0.5 * (ymin + ymax))**(1. / (alpha + 1))
    else:
        ymin, ymax = np.log(ymin), np.log(ymax)
        return np.exp(0.5 * (ymin + ymax))


NBINS = 1000


class TappedPower:
    """
    alpha: slope for m->0
    beta: slope for m-> inf, usually beta <0. It must satisfy alpha > beta
    mstar: turning point mass
    flex: turning region size, larger flex, smaller turning radius
    mmin: min mass in consideration
    mmax: max mass in consideration
    Examples
    --------
    p = TappedPower(1, -2)
    m = np.logspace(-1, 2, 100)
    loglog(m, p.pdf(m))
    loglog(m, p.cdf(m))
    """

    def __init__(self, alpha, beta, mstar=1, mmin=0.1, mmax=100):
        self.__dict__.update(alpha=alpha, beta=beta, mstar=mstar, mmin=mmin, mmax=mmax)
        self.amp = 1
        self.amp = 1 / self.cdf(mmax)

    def pdf(self, mass):
        amp, alpha, beta, mstar = self.amp, self.alpha, self.beta, self.mstar

        return amp * mass**beta * (1 - np.exp(-(mass / mstar)**(alpha - beta)))

    def cdf(self, mass):
        mmin, mmax = self.mmin, self.mmax

        lgx, dlgx = np.linspace(np.log10(mmin), np.log10(mmax), NBINS, retstep=True)
        x = 10**lgx
        dx = x * dlgx * np.log(10)
        y = (self.pdf(x) * dx).cumsum()

        return np.interp(mass, x, y, left=0)

    def rvs(self, n):
        mmin, mmax = self.mmin, self.mmax

        lgx, dlgx = np.linspace(np.log10(mmin), np.log10(mmax), NBINS, retstep=True)
        x = 10**lgx
        dx = x * dlgx * np.log(10)
        y = (self.pdf(x) * dx).cumsum()

        a = np.random.rand(n)
        return np.interp(a, y, x)


class TwoPower:
    """
    alpha: slope for m->0
    beta: slope for m-> inf, usually beta <0. It must satisfy alpha > beta
    mstar: turning point mass
    flex: turning region size, larger flex, smaller turning radius
    mmin: min mass in consideration
    mmax: max mass in consideration
    Examples
    --------
    p = TwoPower(1, -2)
    m = np.logspace(-1, 2, 100)
    loglog(m, p.pdf(m))
    loglog(m, p.cdf(m))
    """

    def __init__(self, alpha, beta, flex=4, mstar=1, mmin=0.1, mmax=100):
        self.__dict__.update(alpha=alpha, beta=beta, flex=flex, mstar=mstar, mmin=mmin, mmax=mmax)
        self.amp = 1
        self.amp = 1 / self.cdf(mmax)

    def pdf(self, mass):
        amp, alpha, beta, flex, mstar = self.amp, self.alpha, self.beta, self.flex, self.mstar

        return amp * mass**alpha * (1 + (mass / mstar)**flex)**((beta - alpha) / flex)

    def cdf(self, mass):
        mmin, mmax = self.mmin, self.mmax

        lgx, dlgx = np.linspace(np.log10(mmin), np.log10(mmax), NBINS, retstep=True)
        x = 10**lgx
        dx = x * dlgx * np.log(10)
        y = (self.pdf(x) * dx).cumsum()

        return np.interp(mass, x, y, left=0)

        # from scipy.special import hyp2f1
        # x, a, b, c, d = mass, alpha, beta, flex, mstar
        # cdf_unorm = (x**(1 + a) * hyp2f1((1 + a) / c, (a - b) / c, (1 + a + c) / c, -(x / d)**c)) / (1 + a)

    def rvs(self, n):
        mmin, mmax = self.mmin, self.mmax

        lgx, dlgx = np.linspace(np.log10(mmin), np.log10(mmax), NBINS, retstep=True)
        x = 10**lgx
        dx = x * dlgx * np.log(10)
        y = (self.pdf(x) * dx).cumsum()

        a = np.random.rand(n)
        return np.interp(a, y, x)


def cal_MFweight(Mass_all, IMF_form, IMF_args):

    if IMF_form == 'Salpeter':
        alpha = IMF_args[0]
        mass_P = np.diff(pow_cdf(Mass_all, alpha, Mass_all.min(), Mass_all.max()))

    elif IMF_form == 'TapedPower':
        alpha, beta, mstar = IMF_args
        mf_ = TappedPower(alpha, beta, mstar, Mass_all.min(), Mass_all.max())
        mass_P = np.diff(mf_.cdf(Mass_all))

    elif IMF_form == 'TwoPower':
        alpha, beta, flex, mstar = IMF_args
        mf_ = TwoPower(alpha, beta, flex, mstar, Mass_all.min(), Mass_all.max())
        mass_P = np.diff(mf_.cdf(Mass_all))

    w_i = mass_P / sum(mass_P)
    return w_i

def cal_q_weight(q_alpha, q_bins):

    q_w = np.diff(pow_cdf(q_bins, q_alpha, min(q_bins), max(q_bins)))
    return q_w


def cal_model_weight(model_mass, IMF_form, IMF_args, f_b, q_alpha, q_bins):
    w_mf = cal_MFweight(model_mass, IMF_form, IMF_args)
    w_q = cal_q_weight(q_alpha, q_bins)
    w_binary = np.hstack([1 * (1 - f_b), w_q * f_b])
    w_i = (w_binary.reshape(-1, 1) * w_mf).ravel()
    return w_i


def cal_av_model(av, _iso_model, Av_grid):
    if av == 0:
        av_model = _iso_model[0]
    else:
        av_i = Av_grid.searchsorted(av)
        _Av1 = Av_grid[av_i]
        _Av0 = Av_grid[av_i - 1]
        c_av = (av - _Av0) / (_Av1 - _Av0)
        av_model = (_iso_model[av_i] - _iso_model[av_i - 1]) * c_av + _iso_model[av_i - 1]
    return av_model


#==============================================================================
# FS model

def assert_odd(a):
    if a % 2:
        return a
    else:
        return a + 1


def gener_pmatrix(xerr, yerr, d_x, d_y, sig=6):

    from scipy.stats import norm
    #assert sig < 15

    nbins_x = np.int32((assert_odd(np.ceil(2 * sig * xerr / d_x)) - 1) / 2)
    nbins_y = np.int32((assert_odd(np.ceil(2 * sig * yerr / d_y)) - 1) / 2)

    xbins = np.arange(-nbins_x - 0.5, nbins_x + 1) * d_x
    ybins = np.arange(-nbins_y - 0.5, nbins_y + 1) * d_y

    p_x = np.diff(norm.cdf(xbins, scale=xerr))
    p_y = np.diff(norm.cdf(ybins, scale=yerr))
    p_x[-nbins_x:] = p_x[nbins_x - 1::-1]
    p_y[-nbins_y:] = p_y[nbins_y - 1::-1]
    p = p_x[:, None] * p_y

    return p / (d_x * d_y)


def add_array(M, A, i, j):
    """add A to M s.t. A's center locates at M[i, j]
    """
    mx, my = A.shape
    nx, ny = M.shape
    assert (mx % 2) and (my % 2)
    mx, my = mx // 2, my // 2

    ix0, ix1 = i - mx, i + mx + 1
    iy0, iy1 = j - my, j + my + 1
    ix, iy = ix0, iy0
    ix0 = 0 if ix0 < 0 else ix0
    iy0 = 0 if iy0 < 0 else iy0
    ix1 = nx - 1 if ix1 >= nx else ix1
    iy1 = ny - 1 if iy1 >= ny else iy1

    M[ix0:ix1, iy0:iy1] += A[ix0 - ix:ix1 - ix,
                             iy0 - iy:iy1 - iy]
    return None


def cal_idx(color, mag, color_grid, mag_grid):
    d_color = color_grid[1] - color_grid[0]
    d_mag = mag_grid[1] - mag_grid[0]
    c_idx = np.int32((color - color_grid[0]) / d_color)
    m_idx = np.int32((mag - mag_grid[0]) / d_mag)
    c_idx = c_idx.clip(0, color_grid.size - 2)
    m_idx = m_idx.clip(0, mag_grid.size - 2)
    return c_idx, m_idx


def findknnr(x, y, n_ngb=10):
    X = np.stack([x, y], axis=-1)
    if n_ngb is None:
        n_ngb = int(np.sqrt(len(x)))
    rk = KDTree(X, leaf_size=20).query(X, n_ngb)[0].T[n_ngb - 1]  # / np.sqrt(n_ngb / np.pi)
    return rk


class Fs_prob:
    def __init__(self, color, mag, color_fs, mag_fs, fs_cerr, fs_merr,
                 d_mag_fs=0.01, d_color_fs=0.01, area_ratio=None,
                 smooth=False, shift=False, c_smooth=1.0):
        keys = ['color', 'mag', 'color_fs', 'mag_fs', 'fs_cerr', 'fs_merr',
                'd_mag_fs', 'd_color_fs', 'area_ratio', 'smooth', 'shift', 'c_smooth']
        for key in keys:

            setattr(self, key, locals()[key])

        if area_ratio is not None:
            self.n_fs_guess = len(color_fs) * area_ratio

        self.color_fs = color_fs[(~np.isnan(fs_merr)) & (~np.isnan(fs_cerr)) &
                                 (~np.isnan(color_fs)) & (~np.isnan(mag_fs))]
        self.mag_fs = mag_fs[(~np.isnan(fs_merr)) & (~np.isnan(fs_cerr)) &
                             (~np.isnan(color_fs)) & (~np.isnan(mag_fs))]
        self.fs_merr = fs_merr[(~np.isnan(fs_merr)) & (~np.isnan(fs_cerr)) &
                               (~np.isnan(color_fs)) & (~np.isnan(mag_fs))]
        self.fs_cerr = fs_cerr[(~np.isnan(fs_merr)) & (~np.isnan(fs_cerr)) &
                               (~np.isnan(color_fs)) & (~np.isnan(mag_fs))]

        self.gene_fs_pdens(smooth, shift, c_smooth)
        fsc_idx, fsm_idx = cal_idx(self.color, self.mag, self.color_grid_fs, self.mag_grid_fs)
        self.fs_prob = self.cal_fs_prob(fsc_idx, fsm_idx)

    def gene_fs_pdens(self, smooth, shift, c_smooth, n_err=6):
        if shift is True:
            color = self.color_fs + np.random.randn(len(self.color_fs)) * self.fs_cerr
            mag = self.mag_fs + np.random.randn(len(self.mag_fs)) * self.fs_merr
        else:
            color = self.color_fs
            mag = self.mag_fs

        self.mag_grid_fs = np.arange(mag.min() - max(self.fs_merr) * n_err - 10 * self.d_mag_fs,
                                     mag.max() + max(self.fs_merr) * n_err + 10 * self.d_mag_fs,
                                     self.d_mag_fs)
        self.color_grid_fs = np.arange(color.min() - max(self.fs_cerr) * n_err - 10 * self.d_color_fs,
                                       color.max() + max(self.fs_cerr) * n_err + 10 * self.d_color_fs,
                                       self.d_color_fs)

        fs_cidx, fs_midx = cal_idx(color, mag, self.color_grid_fs, self.mag_grid_fs)

        pdensity_cmd = np.zeros((len(self.mag_grid_fs) - 1, len(self.color_grid_fs) - 1))

        if smooth is True:
            _rk = findknnr(color, mag)
            m_er = np.sqrt((c_smooth * _rk)**2 + self.fs_merr**2)
            c_er = np.sqrt((c_smooth * _rk)**2 + self.fs_cerr**2)
            for i, j, m_er_, c_er_ in zip(fs_midx, fs_cidx, m_er, c_er):
                P = gener_pmatrix(m_er_, c_er_, self.d_mag_fs, self.d_color_fs, sig=n_err)
                add_array(pdensity_cmd, P, i, j)
        else:
            m_er = self.fs_merr
            c_er = self.fs_cerr
            for i, j, m_er_, c_er_ in zip(fs_midx, fs_cidx, m_er, c_er):
                P = gener_pmatrix(m_er_, c_er_, self.d_mag_fs, self.d_color_fs, sig=n_err)
                add_array(pdensity_cmd, P, i, j)

        pdensity_cmd = pdensity_cmd / len(self.color_fs)
        self.fs_pdens = pdensity_cmd + pdensity_cmd[pdensity_cmd > 0].min()

    def cal_fs_prob(self, color_idx, mag_idx, smooth=False):
        return self.fs_pdens[mag_idx, color_idx]


#==============================================================================

import cyper

code = """
from cython.parallel import prange
import numpy as np
from libc.math cimport sqrt, erf

cpdef double make_sum(double[:, :] obs_arr, double[:, :] mod_arr) nogil:
    cdef:
        int n, m, i, j
        double sum=0.0, x

    n = obs_arr.shape[0]
    m = obs_arr.shape[1]
    with gil:
        assert mod_arr.shape[0] == n, "wrong mod_arr.shape[0]"
        assert mod_arr.shape[1] == m, "wrong mod_arr.shape[1]"
    
    for i in range(n):
        for j in range(m):
            x = mod_arr[i, j]
            if x != 0:
                sum += x * obs_arr[i, j]
            else:
                continue
    return sum
    
cpdef double make_norm(double [:] mag, double [:] wgt, double [:] err, double min, double max) nogil:
    cdef:
        int n, i 
        double sum=0.0, s=sqrt(2.0), p, m, e, d
        
    n = mag.shape[0]        
    with gil:
        assert wgt.shape[0] == n, "wrong wgt shape"
        assert err.shape[0] == n, "wrong err shape"
        
    for i in range(n):
        m = mag[i]
        e = err[i]
        
        d = (max - m) / e
        if d < -5.0:
            continue
        elif d < 5.0:
            p = 0.5 + 0.5 * erf(d / s)
        else:
            d = (min - m) / e
            if d > 5.0:
                continue
            elif d > -5.0:
                p = 0.5 - 0.5 * erf(d / s)
            else:
                p = 1.0
        sum += p * wgt[i]
    return sum

cpdef void add_model(double[:] model_m,
                     double[:] model_c,
                     double[:] model_w,
                     double Mag_min,
                     double Mag_max,
                     double d_mag,
                     double Color_min,
                     double Color_max,
                     double d_c,
                     double[:, :] model_matrix) nogil:
    cdef:
        int n, i, j, k
        double m, c

    n = model_m.shape[0]
    with gil:
        assert model_c.shape[0] == n, "wrong model_c.shape[0]"
        assert model_w.shape[0] == n, "wrong model_w.shape[0]"
        assert model_matrix.shape[0] > int((Mag_max - Mag_min) / d_mag), "wrong model_matrix.shape[0]"
        assert model_matrix.shape[1] > int((Color_max - Color_min) / d_c), "wrong model_matrix.shape[1]"

    for i in range(n):
        m = model_m[i]
        if m <= Mag_min or m >= Mag_max:
            continue
        c = model_c[i]
        if c <= Color_min or c >= Color_max:
            continue

        j = int((m - Mag_min) / d_mag)
        k = int((c - Color_min) / d_c)
        model_matrix[j, k] += model_w[i]
"""

cymod = cyper.inline(code, fast_indexing=True)
add_model = cymod.add_model
make_sum = cymod.make_sum
make_norm = cymod.make_norm
#==============================================================================

def gener_obs_P(mag, color, m_err, c_err, d_m=0.005, d_c=0.005, n_sig=4):

    mag_min = min(mag) - (n_sig + 2) * max(m_err)
    mag_max = max(mag) + (n_sig + 2) * max(m_err)
    color_min = min(color) - (n_sig + 2) * max(c_err)
    color_max = max(color) + (n_sig + 2) * max(c_err)
    norm_mag_min = min(mag)
    norm_mag_max = max(mag)
    mag_grid = np.arange(mag_min, mag_max, d_m)
    color_grid = np.arange(color_min, color_max, d_c)

    m_i0 = np.int32((mag - n_sig * m_err - mag_grid[0]) / d_m)
    m_i1 = np.int32((mag + n_sig * m_err - mag_grid[0]) / d_m)
    c_i0 = np.int32((color - n_sig * c_err - color_grid[0]) / d_c)
    c_i1 = np.int32((color + n_sig * c_err - color_grid[0]) / d_c)

    table = []
    for i in range(len(mag)):
        mag_grid_ = mag_grid[m_i0[i]:m_i1[i] + 2]
        color_grid_ = color_grid[c_i0[i]:c_i1[i] + 2]
        p_m = np.diff(normal.cdf(mag_grid_, mag[i], m_err[i])) / d_m
        p_c = np.diff(normal.cdf(color_grid_, color[i], c_err[i])) / d_c
        p = p_m.reshape(-1, 1) * p_c
        table.append(p)
    obs_pts = table, m_i0, m_i1, c_i0, c_i1, mag_grid, color_grid, d_m, d_c, mag_min, mag_max, color_min, color_max, norm_mag_min, norm_mag_max
    return namedtuple("obs_pts", "obs_matrix, m_i0, m_i1, c_i0, c_i1, mag_grid, color_grid, d_m, d_c, mag_min, mag_max, color_min, color_max, norm_mag_min, norm_mag_max")(*obs_pts)


class MiMO:
    """
    """

    def __init__(self, model_path_s):

        self.load_model(model_path_s)

    def load_model(self, model_path_s):

        with h5py.File(model_path_s, 'r') as fp:
            self.iso_model = fp['iso_model'][...]
            self.iso_model_mass = fp['iso_model_mass'][...]
            self.binary_q_bins = fp['q_bins'][...]
            self.Av_grid = fp['Av_grid'][...]
            self.feh_grid = fp['feh_grid'][...]
            self.age_grid = fp['logAge_grid'][...]

        self.feh_grid_min = self.feh_grid.min()
        self.d_feh = self.feh_grid[1] - self.feh_grid[0]
        self.age_grid_min = self.age_grid.min()
        self.d_age = self.age_grid[1] - self.age_grid[0]
#         self.model_err = np.full_like(model_mag00, 0.02)

    def cl_p(self, obs_pts, age, feh, dm, Av, IMF_form, IMF_args, f_b, q_alpha,
             fs_prob=None, f_fs=None, cal_single=False):

        feh_i = np.int32((feh - self.feh_grid_min) / self.d_feh)
        age_i = np.int32((age - self.age_grid_min) / self.d_age)

        age_0 = self.age_grid[age_i]
        age_1 = self.age_grid[age_i + 1]
        feh_0 = self.feh_grid[feh_i]
        feh_1 = self.feh_grid[feh_i + 1]
        w00, w01, w10, w11 = twod_interp(age, feh, age_0, age_1, feh_0, feh_1)
        wgt_interp = [w00, w01, w10, w11]

        model_matrix = np.zeros((len(obs_pts.mag_grid), len(obs_pts.color_grid)))
        model_norm_value = 0

        for j in range(2):
            for k in range(2):
                model_00 = cal_av_model(Av, self.iso_model[feh_i + k, age_i + j], self.Av_grid)
                model_mag00 = model_00[0] + dm
                model_color00 = model_00[1]
                model_w00_ = cal_model_weight(self.iso_model_mass[feh_i + k, age_i + j], IMF_form, IMF_args, f_b, q_alpha, self.binary_q_bins)
                model_w00 = model_w00_ * wgt_interp[j * 2 + k]

                model_err = np.full_like(model_mag00, 0.02)
                model_norm_value00 = make_norm(model_mag00, model_w00_, model_err, obs_pts.norm_mag_min, obs_pts.norm_mag_max) * wgt_interp[j * 2 + k]

                add_model(model_mag00, model_color00, model_w00, obs_pts.mag_min, obs_pts.mag_max, obs_pts.d_m, obs_pts.color_min, obs_pts.color_max, obs_pts.d_c, model_matrix)
                model_norm_value += model_norm_value00

        cl_prob = np.ones(len(obs_pts.m_i0))
        for i in range(len(obs_pts.m_i0)):
            #         cl_p[i] = sum(obs_matrix[i] * model_matrix[m_i0[i]:m_i1[i]+1, c_i0[i]:c_i1[i]+1])
            cl_prob[i] = make_sum(obs_pts.obs_matrix[i], model_matrix[obs_pts.m_i0[i]:obs_pts.m_i1[i] + 1, obs_pts.c_i0[i]:obs_pts.c_i1[i] + 1])
        cl_prob /= model_norm_value
        if fs_prob is None:
            f_fs = 0
            f_cl = 1
            prob = cl_prob + 1e-20
        else:
            if f_fs is None:
                raise ValueError('if you already gave fs_prob, you must also give us the n_fs!')
            elif f_fs > 1:
                raise ValueError('Number of field stars can not larger than total number of stars!')
            f_cl = 1 - f_fs

            prob = cl_prob * f_cl + fs_prob * f_fs

        ln_lk = np.sum(np.log(prob))

        if cal_single:
            p_member = (cl_prob * f_cl) / prob
            res = ln_lk, np.log(prob), cl_prob, fs_prob, f_cl, p_member
            return namedtuple("ProbResult", "loglh, logp, p_cl, p_fs, f_cl, p_member")(*res)
        else:
            return ln_lk
