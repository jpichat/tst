import math
import numpy as np
from scipy.spatial.distance import squareform, pdist


def granulate_(x, scale):
    N = len(x)
    b = int(np.fix(N / scale))
    u = np.reshape(x[0 : b * scale], (b, scale))
    return np.mean(u, axis=1)


def DaviesBouldin_index(dist_matrix, y, centroid_index):
    def similarity(c):
        """similarity of points in the same cluster"""
        return np.mean(dist_matrix[y == c, :][:, centroid_index[c]])

    def dispersion(ci, cj):
        """dispersion between two clusters"""
        assert ci != cj
        return dist_matrix[centroid_index[ci], centroid_index[cj]]

    def dissimilarity(ci, cj):
        """dissimilarity between two clusters"""
        return (similarity(ci) + similarity(cj)) / dispersion(ci, cj)

    K = np.max(y) + 1  # number of clusters
    DB = 0.0
    for k in range(K):
        DB += np.max([dissimilarity(i, k) for i in range(K) if i != k])
    DB /= K
    return DB


def Silhouette_index(dist_matrix, y):
    def silhouette_(c):
        """Silhouette index of a cluster"""
        # average distance between sample p and all the remaining elements assigned to the same cluster
        a = np.mean(dist_matrix[y == c, :][:, y == c], axis=1)
        n_c = a.shape[0]  # number of points in cluster c
        # minimum average distance between each sample of the cluster c to other clusters
        b = np.min(
            np.stack(
                [np.mean(dist_matrix[y == c, :][:, y == c2], axis=1) for c2 in range(K) if c2 != c],
                axis=1,
            ),
            axis=1,
        )
        s = (b - a) / np.maximum(a, b)
        return np.sum(s) / n_c

    K = np.max(y) + 1  # number of clusters
    # Global silhouette index
    return np.mean([silhouette_(c) for c in range(K)])


def distortion(x, y, centroids):
    d = 0
    for k in range(centroids.shape[0]):
        # distortion of cluster k
        d += np.sum(np.square(x[y == k, :] - centroids[k, :]))
    return d


# complexity measures


def ApEn(x, m=2, r=None, tau=1, dist_func="chebyshev"):
    """the approximate (multiscale if tau>1) entropy [1] of a 1d series x

    NOTE: it reflects the probability that two sequences which are similar
    for n points remain similar at the next point.

    Args
    ----
    m : int (optional, default: 2)
        length of sequences to be compared

    r : float (optional)
        tolerance for accepting matches (r=0.15 if std(x)=1)

    tau : int (optional, default: 1)
        sampling rate of segments (default: skipping no value)
        for multiscale approximate entropy

    dist_func : str
        distance function is usually the L-inf metric

    References
    ----------
    .. [1] Pincus (1991) Approximate entropy as a measure of system complexity
    .. [2] Pincus (2008) Aproximate entropy as an irregularity measure for financial data
    .. [3] https://en.wikipedia.org/wiki/Approximate_entropy
    """
    apen = np.zeros(tau)
    X = dict()
    if r is None:
        r = 0.15 * np.std(x)
    if tau > 1:
        for t in range(1, tau + 1):
            X[t] = granulate_(x, t)
    else:
        X[1] = x
    for i, s in enumerate(X.values()):
        A = apen_(s, m, r, dist_func)
        B = apen_(s, m + 1, r, dist_func)
        apen[i] = A - B
    return apen[0] if tau == 1 else apen


def apen_(x, d, r, dist_func="chebyshev"):
    N = len(x)
    X = embed_np(x, d)
    D = pdist(X, dist_func)
    dist = squareform(D)
    C = np.array([sum(i <= r for i in u) for u in dist]) / (N - d + 1)
    return sum(np.log(C)) / (N - d + 1)


def SampEn(x, m=2, r=None, tau: int = 1, dist_func="chebyshev"):
    """sample entropy [1a-1b] (multiscale [2] if tau>1) of a 1d series x

    NOTE: it is the probability that:
    if 2 sets of simultaneous data points of length m have a distance < r, then
    2 sets of simultaneous data points of length m + 1 also have a distance <r

    Args
    ----
    m : int (optional, default: 2)
        length of sequences to be compared

    r : float (optional)
        tolerance for accepting matches (r=0.15 if std(x)=1)

    tau : int (optional, default: 1)
        sampling rate of segments (default: skipping no value)
        for multiscale approximate entropy

    dist_func : str
        distance function is usually the L-inf metric

    References
    ----------
    .. [1a] Richman and Moorman (2000) Physiological time-series analysis using approximate
    entropy and sample entropy
    .. [1b] Lake et al. (2002) Sample entropy analysis of neonatal heart rate variability
    .. [2] Madalena Costa, Ary Goldberger, CK Peng. (2005) Multiscale entropy analysis of
    biological signals
    .. [3] http://en.wikipedia.org/wiki/Sample_Entropy
    """
    assert isinstance(tau, int), "tau must be an integer sampling rate"
    sampen = np.zeros(tau)
    X = dict()
    if r is None:
        r = 0.15 * np.std(x)
    if tau > 1:
        for t in range(1, tau + 1):
            X[t] = granulate_(x, t)
    else:
        X[1] = x
    for i, s in enumerate(X.values()):
        A = sampen_(s, m + 1, r, dist_func)
        B = sampen_(s, m, r, dist_func)
        sampen[i] = -np.log(A / B)
    return sampen[0] if tau == 1 else sampen


def sampen_(x, d, r, dist_func="chebyshev"):
    X = embed_np(x, d)
    D = pdist(X, dist_func)
    S = squareform(D)
    # remove self-similar patterns
    S = S[~np.eye(S.shape[0], dtype=bool)].reshape(S.shape[0], -1)
    return sum([sum(i <= r for i in u) for u in S])


def PermEn(x, m, tau=1, weighted=False):
    """normalised permutation entropy [1] (weighted [3], multiscale if tau>1 [2]) of a series x

    NOTE: non-normalised PermEn ranges between [0, log(d!)]

    Args
    ----
    m : int
        order of permutation entropy (embedding dimension)
        i.e: how much information from the past is used

    tau : int
        sampling rate

    weighted : bool
        see [3], weighted PermEn weights amplitude of fluctuations of d-long segments

    References
    ----------
    .. [1] Bandt and Pompe (2002) Permutation entropy - a natural complexity measure
    for time series
    .. [2] Morabito et al. Multivariate Multi-Scale Permutation Entropy for Complexity
    Analysis of Alzheimer’s Disease EEG
    .. [3] Fadlallah et al. (2013) Weighted-permutation entropy: A complexity measure
    for time series incorporating amplitude information
    """
    X = dict()
    pe = np.zeros(tau)
    if tau > 1:
        for t in range(1, tau + 1):
            X[t] = granulate_(x, t)
    else:
        X[1] = x
    permutations = np.array(list(itertools.permutations(range(m))))
    for k, x in X.items():
        N = len(x)
        c = np.zeros(len(permutations))
        w = np.zeros(N - m + 1)
        for j in range(N - m + 1):
            v = x[j : j + m]
            if weighted:
                w[j] = np.sum((v - np.mean(v)) ** 2) / float(m)
            else:
                w[j] = 1
            pattern = list(np.argsort(v, kind="quicksort")[::-1])
            c[permutations.tolist().index(pattern)] += w[j]
        if weighted:
            p = c[np.nonzero(c)] / float(sum(w))
        else:
            p = c[np.nonzero(c)] / float(N - m + 1)
        # normalise with max entropy
        pe[k - 1] = -sum(p * np.log(p)) / np.log(math.factorial(m))
    return pe[0] if tau == 1 else pe
