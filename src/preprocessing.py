import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib as mpl
from matplotlib import font_manager


colors = ["#0072B2", "#E38035", "#660577", "#433F3F", "#999933"] # colorblind-friendly palette
METRIC1 = "r2_permanova_within_species"  # minimize
METRIC2 = "r2_batch_pc1"                 # minimize
METRICS = [METRIC1, METRIC2]

def use_pub_font(primary="Menlo", fallbacks=("Source Code Pro","Inconsolata","DejaVu Sans Mono")):
    # pick first available from list
    families = {f.name for f in font_manager.fontManager.ttflist}
    chosen = primary if primary in families else next((f for f in fallbacks if f in families), "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": chosen,
        "font.size": 15,          # adjust for your figure size
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 250,
        "pdf.fonttype": 42,       # embed TrueType in PDF
        "ps.fonttype": 42,        # embed TrueType in PS
        "svg.fonttype": "none",   # keep text as text in SVG
        "axes.unicode_minus": False,  # proper minus glyph with some fonts
    })
    return chosen

class SpectralPreprocessor(BaseEstimator, TransformerMixin):
    """
    Steps (in order): MSC -> detrend -> derivative -> SNV -> column-centering.
    Fit-dependent parts: MSC (reference mean), column-centering (column mean),
                         detrend (precomputes Vandermonde pinv on wavelength axis).
    """
    def __init__(self,
                 deriv_order=1, sg_window=21, sg_polyorder=3, use_simple_deriv=True,
                 use_snv=True,
                 use_msc=False,
                 detrend_degree=0,                # 0=off, 1=linear, 2=quadratic
                 center_cols=True,
                 wavelengths=None,                # array-like or None
                 epsilon=1e-12):
        self.deriv_order = deriv_order
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.use_simple_deriv = use_simple_deriv
        self.use_snv = use_snv
        self.use_msc = use_msc
        self.detrend_degree = detrend_degree
        self.center_cols = center_cols
        self.wavelengths = None if wavelengths is None else np.asarray(wavelengths, float)
        self.epsilon = float(epsilon)

        # fitted attributes
        self.ref_mean_ = None
        self.col_mean_ = None
        self._t_axis_ = None
        self._Tmat_ = None
        self._Tpinv_ = None

    # ----- internals -----
    def _prepare_axis_and_detrend(self, p):
        if self.detrend_degree <= 0:
            self._t_axis_ = None
            self._Tmat_, self._Tpinv_ = None, None
            return
        if self.wavelengths is not None:
            t = np.asarray(self.wavelengths, float)
            if t.size != p:
                raise ValueError("wavelengths length does not match X.shape[1]")
        else:
            t = np.arange(p, dtype=float)
        # normalize axis to [0,1] for numerical stability
        t = (t - t.min()) / (t.max() - t.min() + self.epsilon)
        # Vandermonde with increasing powers: [1, t, t^2, ...]
        T = np.vander(t, N=self.detrend_degree+1, increasing=True)  # shape (p, deg+1)
        Tpinv = np.linalg.pinv(T)                                   # (deg+1, p)
        self._t_axis_ = t
        self._Tmat_, self._Tpinv_ = T, Tpinv

    def _apply_msc(self, X):
        if not self.use_msc:
            return X
        R = self.ref_mean_.ravel()
        A = np.vstack([np.ones(R.shape[0]), R]).T      # (p, 2)
        A_pinv = np.linalg.pinv(A)                     # (2, p)
        theta = X @ A_pinv.T                           # (n, 2)
        a = theta[:, 0][:, None]
        b = theta[:, 1][:, None]
        return (X - a) / (b + self.epsilon)

    def _apply_detrend(self, X):
        if self.detrend_degree <= 0:
            return X
        # Trend = (X @ Tpinv^T) @ T^T
        Beta = X @ self._Tpinv_.T                      # (n, deg+1)
        Trend = Beta @ self._Tmat_.T                   # (n, p)
        return X - Trend

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        n, p = X.shape

        # MSC reference on training data
        self.ref_mean_ = X.mean(axis=0, keepdims=True) if self.use_msc else None

        # detrend precompute
        self._prepare_axis_and_detrend(p)

        # ---- compute column mean on TRANSFORMED training data (up to SNV) ----
        if self.center_cols:
            Z = X.copy()
            # 1) MSC
            if self.use_msc:
                R = self.ref_mean_.ravel()
                A = np.vstack([np.ones(R.shape[0]), R]).T
                A_pinv = np.linalg.pinv(A)
                theta = Z @ A_pinv.T
                a = theta[:, 0][:, None]; b = theta[:, 1][:, None]
                Z = (Z - a) / (b + self.epsilon)
            # 2) detrend
            if self.detrend_degree > 0:
                Beta = Z @ self._Tpinv_.T
                Trend = Beta @ self._Tmat_.T
                Z = Z - Trend
            # 3) derivative
            Z = sg_derivative_rows(Z,
                                   window=self.sg_window,
                                   polyorder=self.sg_polyorder,
                                   deriv=(0 if self.deriv_order is None else self.deriv_order),
                                   use_simple_deriv=self.use_simple_deriv)
            # 4) SNV
            if self.use_snv:
                Z = snv(Z)
            # store mean of the transformed TRAIN data
            self.col_mean_ = Z.mean(axis=0, keepdims=True)
        else:
            self.col_mean_ = None

        return self

    def transform(self, X):
        X = np.asarray(X, float)
        Z = X

        # 1) MSC
        if self.use_msc:
            R = self.ref_mean_.ravel()
            A = np.vstack([np.ones(R.shape[0]), R]).T
            A_pinv = np.linalg.pinv(A)
            theta = Z @ A_pinv.T
            a = theta[:, 0][:, None]; b = theta[:, 1][:, None]
            Z = (Z - a) / (b + self.epsilon)

        # 2) detrend
        if self.detrend_degree > 0:
            Beta = Z @ self._Tpinv_.T
            Trend = Beta @ self._Tmat_.T
            Z = Z - Trend

        # 3) derivative
        Z = sg_derivative_rows(Z,
                               window=self.sg_window,
                               polyorder=self.sg_polyorder,
                               deriv=(0 if self.deriv_order is None else self.deriv_order),
                               use_simple_deriv=self.use_simple_deriv)

        # 4) SNV
        if self.use_snv:
            Z = snv(Z)

        # 5) column centering using TRAIN mean of the transformed data
        if self.center_cols and (self.col_mean_ is not None):
            Z = Z - self.col_mean_

        return Z

    # keep the safe helpers
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def fit_transform_pair(self, X_train, X_test=None, y=None):
        Xtr = self.fit(X_train, y).transform(X_train)
        Xte = None if X_test is None else self.transform(X_test)
        return Xtr, Xte 
    
# --- helpers ---
def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def _sg_coeffs(window, polyorder, deriv):
    assert window % 2 == 1, "window must be odd"
    half = window // 2
    x = np.arange(-half, half+1, dtype=float)
    A = np.vstack([x**k for k in range(polyorder+1)]).T
    pinv = np.linalg.pinv(A)
    from math import factorial
    coeff = pinv[deriv] * factorial(deriv)
    return coeff

def sg_derivative_rows(X, window=21, polyorder=3, deriv=1, use_simple_deriv=False):
    if deriv is None or deriv == 0:
        return X.astype(float, copy=True)
    if use_simple_deriv:
        dX = np.empty_like(X, dtype=float)
        dX[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / 2.0
        dX[:, 0] = X[:, 1] - X[:, 0]
        dX[:, -1] = X[:, -1] - X[:, -2]
        return dX
    window = int(window) + (1 - int(window) % 2)
    if polyorder < deriv:
        raise ValueError("polyorder must be ≥ deriv")
    coeff = _sg_coeffs(window, polyorder, deriv)
    half = window // 2
    Xpad = np.pad(X, ((0,0),(half,half)), mode="reflect")
    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        seg = Xpad[:, j:j+window]
        out[:, j] = np.dot(seg, coeff)
    return out

