import warnings
import math
import numpy as np
import pandas as pd
import typing as t
import numba as nb
from typing import List, Union
from scipy import signal
from scipy.optimize import curve_fit
from scipy.linalg import hankel
from scipy.interpolate import LSQUnivariateSpline
from scipy.stats.mstats import theilslopes
from scipy.stats import shapiro, levene, gaussian_kde
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import adfuller, kpss
from itertools import compress
from statsmodels.tsa.stattools import acf


def weighted_correlation(
    arr: np.ndarray, window_size: int, local_constraint=None, condensed=False
):
    """weighted-correlations matrix (used in SSA)

    Args
    ----
    arr : np.ndarray
        series of reconstructed components to be compared (along rows)

    window_size : int
        window size

    local_constraint : int (optional, default: None)
        umber of neighbouring components to compare against (if one wants to restrict comparison to a smaller n)

    condensed : Boolean
        whether distance is given in condensed form (vectorised upper triangle of square distance matrix)

    Returns
    -------
    C : np.ndarray
        weighted-correlations matrix
    """
    n, N = arr.shape
    r, c = np.triu_indices(n, 1)
    w = np.array([min(i, window_size, N + 1 - i) for i in range(1, N + 1)])
    w_ip = np.sum(w * arr[r] * arr[c], axis=1)
    w_norm1 = np.sqrt(np.inner(w, np.power(arr[r], 2)))
    w_norm2 = np.sqrt(np.inner(w, np.power(arr[c], 2)))
    C = np.abs(w_ip / (w_norm1 * w_norm2))

    if local_constraint is not None:
        assert isinstance(local_constraint, int)
        ind_zeros = np.where(np.abs(r - c) > local_constraint)[0]
        C[ind_zeros] = 0

    if condensed:
        return C
    else:
        return squareform(C) + np.eye(n)


def granulate(x: np.ndarray, max_len: int):
    """mean down-sampling of input series 'x'

    Args
    ----
    x : np.array
        time series

    max_len : int
        maximum output length (inclusive)
    """
    N = len(x)
    q = int(np.ceil(N / max_len))
    b = int(np.fix(N / q))
    u = np.reshape(x[0 : b * q], (b, q))
    return np.mean(u, axis=1)


def dict_diff(old: dict = None, new: dict = None) -> dict:
    """returns changes/additions in new compared to old"""
    if not bool(new):
        return {}

    if not bool(old):
        return new if new else {}

    assert isinstance(old, dict), type(old)
    assert isinstance(new, dict), type(new)

    diff = {}
    for k, v in new.items():
        if v != old.get(k):
            diff.update({k: v})

    return diff


def dot_avg_ma(H: np.ndarray, P: np.ndarray):
    """diagonal-averaging (avg) of main anti-diagonal (ma)
    of transformed matrix in the original variables

    Args
    ----
    H : np.ndarray
        [LxL] (hankel) matrix

    P : np.ndarray
        [LxL] projection matrix

    Returns
    -------
    mean of the main anti-diagonal
    """
    h, w = H.shape
    assert h == w, "matrix is not square"

    H = H.ravel()
    P = np.fliplr(P).ravel()

    return np.dot(H, P) / h


def dot_avg_at(H: np.ndarray, P: np.ndarray, which="upper"):
    """diagonal-averaging (avg) of k-antidiagonals of upper/lower antitriangle (at)
    of transformed matrix in the original variables

    Args
    ----
    H : np.ndarray
        [LxL] (hankel) matrix
    P : np.ndarray
        [LxL] projection matrix
    which : str (optional, default: "upper")
        whether one is interested in upper or lower antitriangle elements

    Returns
    -------
    [1xL-1] means of k-antidiagonals of upper/lower antitriangle

    Use
    ---
    mlcore.transform.decomposition.SSA
    """
    K, L = H.shape
    assert K == L, "matrix is not square"

    if which == "upper":
        R = np.flip(np.triu_indices(L, 1)[0], 0)
        C = np.tril_indices(L, -1)[1]
        r = range(-K + 1, -K + L)

    elif which == "lower":
        R = np.flip(np.tril_indices(L, -1)[0], 0)
        C = np.triu_indices(L, 1)[1]
        r = range(1, L)

    else:
        raise ValueError("unknown parameter")

    res = np.zeros((L, L), dtype=np.float64)
    res[R, C] = np.einsum("ij,ij->j", H[:, R], P[:, C])
    res = np.flipud(res)

    return np.array([np.mean(np.diag(res, j)) for j in r])


def connected_components(l: list, step=1):
    """find groups of step-consecutive/adjacent numbers (most frequently unique
    indices, but not necessarily) in ``l``

    Returns
    -------
    list_of_groups : list
        sorted list of lists of consecutive numbers

    NOTE: the function does not assume that numbers in ``l`` are ordered or unique

    Examples
    --------
    > l = [1, 3, 10, 4, 14, 6, 7, 2, 3, 4, 8, 9, 13, 13, 14]
    > connected_components(l)
    [[1, 2, 3, 4], [6, 7, 8, 9, 10], [13, 14]]
    """
    group = []
    list_of_groups = [group]
    expected_num = None
    for v in set(l):
        if (v == expected_num) or (expected_num is None):
            group.append(v)
        else:
            group = [v]
            list_of_groups.append(group)
        expected_num = v + step
    return list_of_groups


@nb.jit(nopython=True)
def max_consecutive_nans(arr, nskip=0):
    """find max consecutive nans in 2d arrays
    x=[[np.nan, 1, 2, np.nan, np.nan, 2, np.nan]] -> [2]
    x=[[np.nan, np.nan, 2, 3, np.nan, 2, np.nan],
       [np.nan, 2, 3, np.nan, np.nan, np.nan, 1]] -> [2,3]

    adapted from: https://stackoverflow.com/a/41722059
    """
    out = []
    arr = arr[:, nskip:]
    row = 0
    while row < arr.shape[0]:
        max_ = 0
        idx = 0
        current = 0
        while idx < arr[row].size:
            while idx < arr[row].size and math.isnan(arr[row, idx]):
                current += 1
                idx += 1
            if current > max_:
                max_ = current
            current = 0
            idx += 1
        out.append(max_)
        row += 1
    return out


@nb.jit(nopython=True)
def max_consecutive_value(arr, nskip=0, val=0):
    out = []
    arr = arr[:, nskip:]
    row = 0
    while row < arr.shape[0]:
        max_ = 0
        idx = 0
        current = 0
        while idx < arr[row].size:
            while idx < arr[row].size and arr[row, idx] == val:
                current += 1
                idx += 1
            if current > max_:
                max_ = current
            current = 0
            idx += 1
        out.append(max_)
        row += 1
    return out


@nb.jit(nopython=True)
def max_consecutive_numeric(arr, nskip=0):
    """find max consecutive numbers in 2d arrays

    .. example::

        x=[[np.nan, 1, 2, np.nan, np.nan, np.nan, np.nan]] -> [2]

        x=[[np.nan, np.nan, 2, 3, 3, 2, np.nan],
       [np.nan, 2, 3, 1, np.nan, np.nan, 1]] -> [4,3]
    """
    out = []
    arr = arr[:, nskip:]
    row = 0
    while row < arr.shape[0]:
        max_ = 0
        idx = 0
        current = 0
        while idx < arr[row].size:
            while idx < arr[row].size and not math.isnan(arr[row, idx]):
                current += 1
                idx += 1
            if current > max_:
                max_ = current
            current = 0
            idx += 1
        out.append(max_)
        row += 1
    return out


@nb.jit(nopython=True)
def constant_columns(arr: np.ndarray) -> list:
    assert arr.ndim == 2
    out = []
    for c in range(arr.shape[1]):
        if np.all(arr[:, c] == arr[0, c]):
            out.append(c)
    return out


def alpha_(a, n, reverse=False):
    """non-normalized exponential weights"""
    alpha_ = [a * np.power(1 - a, i) for i in range(n)]
    if reverse:
        return alpha_[::-1]
    else:
        return alpha_


def segment(
    df: pd.DataFrame,
    field: str,
    dates=None,
    n_missing_values: int = 20,
    min_tts_length: int = 60,
    reference_date_range=None,
    verbose: bool = False,
):
    """identifies "significant" clusters of missing values in a single 1d series, truncates (remaining punctual missing
        values, if any, are set to 0) and optionally realigns it wrt a reference timescale

    Args
    ----
    dates : DateIndex, optional
        set of dates (df.index)

    n_missing_values : int
        maximum number of consecutive missing values to
        tolerate (could be changed to a percentage value instead?)

    min_tts_length : int
        minimum number of observations in the truncated time series (tts)

    reference_date_range : DateIndex, optional
        reference time scale

    Notes
    -----
    - trims leading/ending NaNs (due to time alignment)
    - fills actual missing values with 0s
    - picks values left after truncating significant clusters (if any)
    - aligns back with reference timescale, i.e. pre/postpends NaNs (if any)
    - "significant" in the sense: groups of size greater than n_missing_values.
    - the resulting truncated time series (tts) must have more than min_tts_length
        observed time points left (for meaningful modelling)
    - if more than 1 significant cluster exists, only the last one is considered
        since the goal is to forecast the right-most end of x
    - returns None if too few observations are left in the truncated series
    """
    # removes leading or ending NaNs (added for alignment) from series and dates
    x, date_idx = trim_nans(df, field, dates=dates)
    # finding NaNs corresponding to missing values
    not_nans_where = np.argwhere(~np.isnan(x)).reshape(-1)
    nans_indices = np.array(sorted(set(np.arange(len(x))) - set(not_nans_where)))
    # grouping adjacent indices into clusters
    nans_clusters = connected_components(nans_indices)
    # shortlist last index of significant clusters of missing values
    significant_indices = [u[-1] for u in nans_clusters if len(u) > n_missing_values]

    if len(significant_indices) > 0:
        s_idx = significant_indices[-1] + 1

        if len(x[s_idx:]) > min_tts_length:
            if verbose:
                warnings.warn(f">>> {field} was truncated.")

            if dates is not None:
                ts_truncated = pd.DataFrame(
                    x[s_idx:], columns=[field], index=date_idx[s_idx:]
                ).fillna(0)
            else:
                ts_truncated = pd.DataFrame(x[s_idx:], columns=[field]).fillna(0)

            if dates is not None and reference_date_range is not None:
                return ts_truncated.reindex(index=reference_date_range)
            else:
                return ts_truncated
        else:
            if verbose:
                warnings.warn(
                    f">>> {field} has too few observations left "
                    f"({len(x[s_idx:])}<min_tts_length ({min_tts_length}))."
                )
            return None
    else:
        if len(x) > min_tts_length:
            if verbose:
                warnings.warn(">>> {field} was truncated.")
            ts = pd.DataFrame(x, columns=[field], index=date_idx).fillna(0)
            if dates is not None and reference_date_range is not None:
                return ts.reindex(index=reference_date_range)
            else:
                return ts
        else:
            if verbose:
                warnings.warn(">>> {field} has too few observations.")
            return None


def trim_nans(df, field, dates=None):
    """removes leading or ending NaNs

    dates : DateIndex
        trims associated dates (df.index)
    """
    a = df[field].values
    acc = np.maximum.accumulate  # pylint: disable=no-member
    m = ~np.isnan(a)
    return (
        a[acc(m) & acc(m[::-1])[::-1]],
        [] if dates is None else list(compress(dates, acc(m) & acc(m[::-1])[::-1])),
    )


def page_matrix_fwd(df: pd.DataFrame, fields: List[str], window_size: int, n_targets=0):
    """splits the time series into `k` non-overlapping `window_size`-long segments (ref: left-end of `x`)

    Returns
    -------
    P : DataFrame
        Page matrix of `x` with left-reference

    Notes
    -----
    - The transform is equivalent to sampling the rows of the associated Hankel matrix
    - By default: `P` is cropped if `window_size` is not a divisor of `len(data)`.
    - if `values_fields` is a list of fields (several time series): 1) the same window size is used for all; 2)
        we assume all time series have same length

    References
    ----------
    .. [1] http://www.publications.pvandenhof.nl/Paperfiles/Damen&etal_S&CL1982.pdf
    .. [2] https://arxiv.org/pdf/1802.09064.pdf
    """
    assert isinstance(fields, list)
    assert window_size > 0
    assert n_targets >= 0

    k = len(df) - (window_size + n_targets) + 1
    out = []
    output_fields = []

    for k, field in enumerate(fields):
        output_fields.append(
            [
                f"TS{k}_T{i:+}" if i != 0 else "TS{k}_T"
                for i in range(-window_size + 1, n_targets + 1)
            ]
        )
        out.append(
            hankel(df[field], np.zeros(window_size + n_targets))[
                :: window_size + n_targets
            ][:k, :]
        )

    if len(df) % (window_size + n_targets) != 0:
        return pd.DataFrame(
            np.concatenate(out, axis=1)[:-1], columns=sum(output_fields, [])
        )

    else:
        return pd.DataFrame(np.concatenate(out, axis=1), columns=sum(output_fields, []))


def page_matrix_bwd(data: pd.DataFrame, values_fields, window_size: int, n_targets=0):
    """splits the time series into `K` non-overlapping `window_size`-long segments (ref: right-end of `x`)

    Returns
    -------
    P : DataFrame
        Page matrix of `x` with right-reference

    Notes
    -----
    if `window_size` is a divisor of `len(data)`, the result is the same as `forward_Page_matrix`
    """
    if not isinstance(values_fields, list):
        values_fields = [values_fields]
    assert window_size > 0
    assert n_targets >= 0
    K = len(data) - (window_size + n_targets) + 1
    P = []
    output_fields = []
    for k, ts_field in enumerate(values_fields):
        output_fields.append(
            [
                f"TS{k}_T{i:+}" if i != 0 else "TS{k}_T"
                for i in range(-window_size + 1, n_targets + 1)
            ]
        )
        H = hankel(data[ts_field][::-1], np.zeros(window_size + n_targets))[
            :: window_size + n_targets
        ][:K, :]
        if len(data) % (window_size + n_targets) != 0:
            P.append(np.flipud(np.fliplr(H[:-1])))
        else:
            P.append(np.flipud(np.fliplr(H)))
    return pd.DataFrame(np.concatenate(P, axis=1), columns=sum(output_fields, []))


# utils
def Page_matrix_fwd_np(x, L, n_targets=0):
    """splits x into K non-overlapping L-long segments (left-end reference)"""
    K = len(x) - (L + n_targets) + 1
    P = hankel(x, np.zeros(L + n_targets))[:: L + n_targets][:K, :]
    if len(x) % (L + n_targets) != 0:
        return P[:-1]
    else:
        return P


def Page_matrix_bwd_np(x, L, n_targets=0):
    """splits `x` into `K` non-overlapping `L`-long segments (ref: right-end of `x`)

    Args
    ----
    x : ndarray
        input time series
    L : int
        window size
    n_targets : int
        number of targets

    Notes
    -----
    same as `backward_Page_matrix` except the input is a `numpy` array
    """
    K = len(x) - (L + n_targets) + 1
    P = hankel(x[::-1], np.zeros(L + n_targets))[:: L + n_targets][:K, :]
    if len(x) % (L + n_targets) != 0:
        return np.flipud(np.fliplr(P[:-1]))
    else:
        return np.flipud(np.fliplr(P))


def nan_clusters(df: pd.DataFrame, field: str):
    """finds indices of clusters of nans in a 1d series"""
    nans_indices = df[field].index[df[field].apply(np.isnan)].tolist()
    return connected_components(nans_indices)


def is_nonstationary(df: pd.DataFrame, field: str):
    """check whether series is over/under-differenced using ADF, KPSS and
    Levene test (weak stationarity might be enough)
    """
    # test for normality
    pval_shapiro = shapiro(df[field].values)[1]

    # test constant variance
    center = "median" if pval_shapiro < 0.05 else "mean"
    pval_levene = levene(*list(get_segments(df[field].values, n=50)), center=center)[1]

    # test stationarity
    pval_adf = adfuller(df[field].values)[1]
    pval_kpss = kpss(df[field].values)[1]

    return pval_adf > 0.05 and pval_kpss < 0.05 and pval_levene < 0.05


def suggest_diff_order(df: pd.DataFrame, field: str):
    if is_nonstationary(df, field):
        return 1
    else:
        return 0


def is_dummy(data_source, field: str, max_ordinal: int = 3) -> list:
    """
    Simple (dumb) dummy guesser that checks whether a subset of a fields' values are all
    in [0, max_ordinal-1]
    """
    # arbitrary so we don't check all values (gain of time if series is long, but less accurate)
    df = data_source.inmem([field]).sample(
        n=min(500, len(data_source)), random_state=42
    )

    if all([x in set(np.arange(max_ordinal)) for x in set(df[field].values)]):
        return True

    return False


def get_segments(x, n=100, l=None, seed=None):
    """extracts n sublists of (random) lengths (15<l<Card(x)/3)"""
    s = 42 if seed is None else seed
    np.random.seed(s)
    for _ in range(n):
        p = np.random.randint(15, int(len(x) / 3)) if l is None else l
        q = np.random.randint(0, p - 1)
        yield x[q : q + p]


# using autocorrelation
def autodetect_window_autocorrelation(
    x: np.array,
    min_period: int = 4,
    corr_thr: float = 0.5,
    alpha: float = 0.05,
    trials: int = 100,
    verbose: bool = False,
):
    """estimate window size using auto-correlation function

    Args
    ----
    x : ndarray
        1d time series

    @author: Yury Tsoy
    """
    len_4 = int(len(x) / 4)
    lags = []
    for _ in range(trials):
        cur_idx = np.random.randint(len_4)
        cur_xs = x[cur_idx:]
        lag = autodetect_lag_acf(cur_xs, min_period, corr_thr, alpha)
        lags.append(lag)
    if verbose:
        warnings.warn(f">>> lag={max(lags)}")

    return int(max(lags))


def autodetect_lag_acf(x, min_period, corr_thr=0.5, alpha=0.05):
    n = np.isnan(x).sum()
    acf_vals, conf_int = acf(
        x, alpha=alpha, nlags=min(1000, len(x) - (n + 1)), missing="drop", fft=False
    )
    upper_bound = conf_int[:, 1] - acf_vals
    w = np.where((np.abs(acf_vals) >= corr_thr) & (np.abs(acf_vals) > upper_bound))[0]
    if len(w) == 0:
        # when no significant lag can be found, just use 1 lag
        return 1
    res = max(w[-1], min_period)
    res = min(res, min(len(x) / 3, 512))
    return res


def autodetect_window_periodogram(x: np.array, min_period=4):
    """estimate window size (period) of time series x using peaks of its spectral density
    (estimated using Welch’s method)

    Args
    ----
    x : ndarray
        1d time series
    min_period : int
        lower bound on the window size to estimate

    Returns
    -------
    period : int
        weighted average of candidate periods

    Notes
    -----
    - `x` should first be detrended in order for the periodogram to score periods well: coarse median filtering
        is used by default
    - Rob J. Hyndman recommends `max_window_size=len(x)/2`
    """
    max_period = min(len(x) // 3, 512)
    x_detrended = detrend(x, which="median", period=max_period)[0]
    x_detrended[np.isnan(x_detrended)] = 0.0  # replacing x with window mean/median

    # peak powers and associated periods
    peaks = periodogram_peaks(x_detrended, min_period=min_period)
    if peaks is None:
        return max_period

    periods, scores = zip(*peaks)
    period = int(round(np.average(periods, weights=scores)))

    return period


class FilterType:
    MEAN = "mean"
    MEDIAN = "median"
    LINE = "line"
    SPLINE = "spline"
    POLYNOMIAL = "polynomial"
    POWER_LAW = "powerlaw"
    WHITE_TOPHAT = "white_tophat"
    BLACK_TOPHAT = "black_tophat"
    MORPHO = "morpho"


def detrend(x: np.array, which="spline", period=None):
    """detrend time series x

    Args
    ----
    x : ndarray
        1d time series
    period : optional, int
        whether the period for detrending is known

    .. note:: The function `straighten_edges`: 1) allows to deal with undesirable boundary effects that may occur when
        using spline, polynomial or power law fitting (by fitting straight lines at both ends); 2) should be used by
        default when applying mean/median filtering
    """
    if which is None:
        return np.ones(len(x)) * np.mean(x)

    if period is None:
        period = autodetect_window_periodogram(x)  # rough period estimate
    # ensuring odd window (symmetric around centre)
    window = 2 * (int(2 * period) // 2) + 1

    if which == FilterType.MEAN:
        trend = straighten_edges(mean_filter(x, window), window)
    elif which == FilterType.MEDIAN:
        trend = straighten_edges(median_filter(x, window), window)
    elif which == FilterType.LINE:
        trend = line_fit(x, window)
    elif which == FilterType.SPLINE:
        n_segments = len(x) // (2 * window) + 1
        trend = straighten_edges(spline_fit(x, n_segments), window)
    elif which == FilterType.POLYNOMIAL:
        trend = straighten_edges(polynomial_fit(x), window)
    elif which == FilterType.POWER_LAW:
        trend = straighten_edges(power_law_fit(x), window)
    elif which == FilterType.WHITE_TOPHAT:
        trend = opening(x, window)
    elif which == FilterType.BLACK_TOPHAT:
        trend = (-1) * closing(x, window)
        x = (-1) * x
    elif which == FilterType.MORPHO:
        trend = morphological_approximation(x, window)
    else:
        raise ValueError("unknown smoothing")
    return x - trend, trend


def periodogram_peaks(
    x: np.array, min_period: int = 4, max_period: int = None, thresh=0.9
):
    """use a modified periodogram (Welch, 1967) to estimate high scoring periods of `x`
    (i.e. above `thresh` of the highest peak)

    Args
    ----
    x : ndarray
        1d time series

    .. note:: x should have no trend for better periods estimations
    """
    periods, power = get_periods(x, min_period, max_period)
    if np.all(np.isclose(power, 0.0)):  # all frequencies have ~0 power (no periodicity)
        return None
    keep = np.where(power > power.max() * thresh)[0]

    return zip(periods[keep], power[keep]) if len(keep) else None


def get_periods(x, min_period=4, max_period=None):
    """get periods and their spectral powers. Welch's method [1,2] is used
    to estimate the power spectral density of `x`

    Args
    ----
    x : ndarray
        1d time series

    .. note:: default overlap between segments (Welch): 50% overlap (no overlap = Bartlett's method)

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Welch%27s_method
    .. [2] Welch (1967) The use of fast Fourier transform for the estimation of
    power spectra: a method based on time averaging over short, modified periodograms
    """
    # min number of cycles of data when setting FFT window (could be 2)
    min_fft_cycles = 3.0

    # max period when setting FFT window
    max_fft_period = 512

    if max_period is None:
        max_period = int(min(len(x) / min_fft_cycles, max_fft_period))

    segment_length = min(2 * max_period, len(x) // 2)  # FFT window

    # estimate power spectral density
    freqs, power = signal.welch(x, scaling="spectrum", nperseg=segment_length)
    periods = np.array([int(round(1.0 / freq)) for freq in freqs[1:]])
    power = power[1:]
    idx = 1

    while idx < len(periods):
        if periods[idx] == periods[idx - 1]:
            # update max corresponding to duplicate periods
            power[idx - 1] = max(power[idx - 1], power[idx])
            periods = np.delete(periods, idx)  # delete duplicate
            power = np.delete(power, idx)
        else:
            idx += 1
    # discard the artifact at freq=1/segment_length
    power[periods == segment_length] = 0.0

    # keep periods/powers within [min,max]
    keep_min = len(periods[periods >= max_period]) - 1
    keep_max = len(periods[periods < min_period])
    periods = periods[keep_min:-keep_max]
    power = power[keep_min:-keep_max]

    return periods, power


def straighten_edges(x: np.array, window_size: int):
    """replace both ends of `x` (`window_size//2` terms) by straight
    line fits using Theil-Sen estimator (on a full window at each end)

    Args
    ----
    x : ndarray
        1d time series

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator
    """
    window_size = 2 * (window_size // 2) + 1  # ensuring odd window size
    w = window_size // 2
    mx = np.ma.masked_array(x, mask=np.isnan(x))
    x_straight = np.copy(mx)
    s1 = theilslopes(mx[:window_size])[0]
    s2 = theilslopes(mx[-window_size:])[0]
    x_straight[0:w] = np.arange(-w, 0) * s1 + x[w]
    x_straight[-w:] = np.arange(1, w + 1) * s2 + x[-w - 1]
    return x_straight


# mathematical morphology
def erode(x, window_size):
    """(discrete) functional erosion by a flat structuring element of size `window_size`

    Args
    ----
    x : ndarray
        1d time series
    """
    window_size = 2 * (window_size // 2) + 1  # ensuring odd window size
    w = window_size // 2
    padded_x = np.hstack(([np.nan] * w, x, [np.nan] * w))
    X = embed_np(padded_x, window_size)
    M = np.nanmin(X, axis=1)
    return M


def dilate(x, window_size):
    """(discrete) functional dilation by a flat structuring element of size `window_size`

    Args
    ----
    x : ndarray
        1d time series
    """
    window_size = 2 * (window_size // 2) + 1  # ensuring odd window size
    w = window_size // 2
    padded_x = np.hstack(([np.nan] * w, x, [np.nan] * w))
    X = embed_np(padded_x, window_size)
    M = np.nanmax(X, axis=1)
    return M


def opening(x, window_size):
    """dilation followed by erosion (see `dilate` and `erode` for details)

    Args
    ----
    x : ndarray
        1d time series
    """
    return dilate(erode(x, window_size), window_size)


def closing(x, window_size: int):
    """erosion followed by dilation (see `dilate` and `erode` for details)

    Args
    ----
    x : ndarray
        1d time series
    """
    return erode(dilate(x, window_size), window_size)


def morphological_approximation(x, window_size):
    """average of opening and closing operations

    Args
    ----
    x : ndarray
        1d time series
    """
    return (opening(x, window_size) + closing(x, window_size)) * 0.5


def polynomial_fit(x, k: int = 2):
    """polynomial fit

    Args
    ----
    x : ndarray
        1d time series

    Notes
    -----
    includes line fit (k=1)
    """
    trend = np.poly1d(np.polyfit(np.arange(len(x)), x, k))
    return trend(np.arange(len(x)))


def power_law(x, a, b, s):
    """power law"""
    return a * np.power(x, -s) + b


def power_law_fit(x):
    """power law approximation

    Args
    ----
    x : ndarray
        1d time series
    """
    h = 1, 1, 0
    popt, _ = curve_fit(power_law, np.arange(1, len(x) + 1), x, p0=h)
    trend = power_law(np.arange(1, len(x) + 1), *popt)
    return trend


def spline_fit(x, nsegs=1):
    """fit a cubic spline

    Args
    ----
    x : ndarray
        1d time series
    """
    idx = np.arange(len(x))
    nk = max(2, nsegs + 1)
    knots = np.linspace(idx[0], idx[-1], nk + 2)[1:-2]
    return LSQUnivariateSpline(idx, x, knots)(idx)


def line_fit(x, window_size):
    """fit a line after `median(x)`

    Args
    ----
    x : ndarray
        1d time series
    """
    window_size = 2 * (window_size // 2) + 1  # ensuring odd window size
    half = window_size // 2
    coarse = median_filter(x, window_size)[half:-half]
    slope, _, lower, upper = theilslopes(coarse)
    if lower <= 0.0 and upper >= 0.0:
        filtered = np.zeros(len(x)) + np.nanmedian(x)
    else:
        intercept = np.nanmedian(x) - (len(x) - 1) / 2 * slope
        filtered = slope * np.arange(len(x)) + intercept
    return filtered


def mean_filter(x, window_size):
    """moving average of `x` using `window_size` with its `window_size//2` terms at both ends left unchanged

    Args
    ----
    x : ndarray
        1d time series

    Notes
    -----
    - the same method as in `median_filter` (via a hankel matrix representation of `x`) may be used
    - use `straighten_edges` to deal with boundaries
    - by default, `signal.convolve` automatically chooses direct or Fourier method based on an estimate of which is faster
    """
    window_size = 2 * (window_size // 2) + 1  # ensuring odd window size
    xc = np.copy(x)
    w = window_size // 2
    h = (1.0 / window_size) * np.ones(window_size)
    xc[w:-w] = signal.convolve(x, h, mode="valid")
    return xc


def median_filter(x, window_size):
    """median filtering of `x` using `window_size` with its `window_size//2` terms at both ends left unchanged

    Args
    ----
    x : ndarray
        1d time series

    Notes
    -----
    use `straighten_edges` to deal with boundaries
    """
    window_size = 2 * (window_size // 2) + 1  # ensuring odd window size
    xc = embed_np(x, window_size)
    w = window_size // 2
    M = np.nanmedian(xc, axis=1)
    return np.hstack((x[:w], M, x[-w:]))


def is_constant(x: np.ndarray):
    """checks whether x consists of a unique value"""
    assert x.ndim == 1, f"Expected 1d input but got {x.ndim}d."
    return np.all(x == x[0])


def gkde(
    x: np.ndarray, min_val: int = -1, max_val: int = 1, n: int = 2000, norm: bool = True
):
    """estimate distribution of x via Gaussian Kernel Density Estimation (kde)

    Args
    ----
    x : np.ndarray
        input data (observed)

    n : int
        number of samples

    norm : bool
        whether probabilities sum to 1
    """

    # find kernel bandwidth param and estimate distribution of x using optimal bandwidth
    kde = gaussian_kde(x, bw_method="scott")

    # get samples and probabilities
    s = np.linspace(min_val - 0.2, max_val + 0.2, n).reshape(-1, 1)
    log_prob = kde.logpdf(s.reshape(1, -1))
    p = np.exp(log_prob)

    if norm:
        p = p / np.sum(p)

    return (s.flatten(), p)


def antidiag(arr: np.ndarray, i: int = None, nlags: int = None) -> t.List[np.ndarray]:
    """returns list of antidiagonals of `arr` considering columns [i*nlags, (i+1)*nlags]

    (ie. its `i`-th lagged variable with `nlags`)
    """
    assert arr.ndim == 2
    i = i or 0
    nlags = nlags or arr.shape[1]
    arrf = np.flipud(arr[:, i * nlags : (i + 1) * nlags])
    return [np.diag(arrf, j) for j in range(-arr.shape[0] + 1, nlags)]


def diag_avg(
    x: Union[pd.Series, pd.DataFrame, np.ndarray],
    fields: List[str] = None,
    output_name: str = None,
    n_skip: int = 0,
) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
    """diag average of numpy array / pandas dataframe

    Returns
    -------
    antidiagonally-averaged data
    """
    is_frame = False
    if isinstance(x, pd.Series):
        if output_name:
            return x.rename(output_name)
        return x

    if isinstance(x, pd.DataFrame):
        is_frame = True
        assert fields is not None, "WARN: missing fields, ordering matters!"
        x = x[fields].values
    else:
        assert isinstance(x, np.ndarray)

    x = x[n_skip:, :]
    if x.ndim == 1:
        return x
    else:
        assert x.ndim == 2, "input should be at most 2d!"
        K, L = x.shape

    xf = np.flipud(x)
    xm = np.array([np.nanmean(np.diag(xf, j)) for j in range(-K + 1, L)])

    if is_frame:
        return pd.Series(xm, name=output_name or "diagavg")
    else:
        return xm


def windowed_name(field: str, timestamp: int, suffix=None) -> str:
    suffix = suffix or ""
    if timestamp == 0:
        return f"{field}_T-{timestamp}{suffix}"
    return f"{field}_T{timestamp:+}{suffix}"


def unwindowed_name(windowed_field: str, suffix: str) -> str:
    assert suffix
    assert windowed_name
    return windowed_field.rsplit(suffix)[0]


def embed(
    df: pd.DataFrame,
    fields,
    n_lags: int,
    date_field=None,
    n_targets=0,
    gap=0,
    keep_dims=True,
    full_matrix=False,
    pad_value=np.nan,
    suffix="",
):
    """computes the trajectory matrix of time series in values_field (n_lags-embedding)
    i.e., split it into K overlapping n_lags-long segments

    Args
    ----
    df : DataFrame
        input data (1d series as columns)

    fields : list
        field(s) column name(s)

    date_field: str
        date column name

    n_lags : int
        number of lags

    n_targets : int (optional, default=0)
        number of targets

    gap : int (optional, default=0)
        gap between latest observation and first prediction - usually used with n_targets>0

    keep_dims : Boolean
        whether the output Hankel matrix has the same number of rows as the input series

    full_matrix : Boolean (optional, default: False)
        whether bottom right antitriangular matrix of nans is returned

    Returns
    -------
    dataframe: trajectory matrix [n_samples, (n_lags + gap + n_targets)],
        where n_samples = len(df) - (n_lags + gap + n_targets) + 1
    """
    w = n_lags + gap + n_targets

    assert w > 0, "window size must be strictly positive"
    if len(df) < w:
        raise ValueError(
            f"Chunk is too small to be windowed by {w} "
            f"(n_lags/gap/n_targets: {n_lags}/{gap}/{n_targets})! Choose window size "
            f"below chunk size: {len(df)} (or use a bigger chunk/dataset)"
        )

    if isinstance(fields, str):
        fields = [fields]

    if date_field is not None:
        if isinstance(date_field, list):
            assert len(date_field) == 1
            date_field = date_field[0]
        else:
            assert isinstance(date_field, str)
        # sanity check
        if date_field in fields:
            fields = list(set(fields) - set([date_field]))

    Hs = []
    M = len(df) + n_lags if full_matrix else len(df)
    skip = 0 if keep_dims else w - 1

    for field in fields:
        windowed_fields = [
            windowed_name(field, t, suffix=suffix)
            for t in range(-n_lags + 1, n_targets + gap + 1)
        ]
        Hs.append(
            pd.DataFrame(
                hankel(
                    np.hstack([[pad_value] * (w - 1), df[field]]),
                    r=np.array([pad_value] * (w)),
                )[skip:M],
                columns=windowed_fields,
            )
        )

    if date_field is None:
        return pd.concat(Hs, axis="columns")
    else:
        D = df.loc[skip:, date_field].reset_index(drop=True)
        return pd.concat([D, *Hs], axis="columns")


def embed_np(x, L):
    """computes the trajectory matrix of x (L-embedding)

    Args
    ----
    x : ndarray
        input series
        if it is given k time series of length N, we assumed they are in a [k x N] matrix

    L : int
        window size
    """
    nd = x.ndim
    if nd == 1:
        N = len(x)
    elif nd == 2:
        N = x.shape[1]
    else:
        raise ValueError("does not support dim>2")
    K = N - L + 1
    if nd == 1:
        return hankel(x, np.zeros(L))[:K]
    else:
        l = []
        for x_ in x:
            l.append(hankel(x_, np.zeros(L))[:K])
        return np.hstack(l)
