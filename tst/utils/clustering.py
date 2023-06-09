import itertools
import sys
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.extmath import softmax


def assign_cluster(cov_predict, cov_fit, labels, method="complete", log=False):
    D = []
    for k in set(labels):
        idx = np.argwhere(labels == k)[0]
        d = [logEuclidean_dist(cov_predict, Y, logX=log, logY=log) for Y in cov_fit[idx]]
        if method == "single":
            D.append(min(d))
        elif method == "complete":
            D.append(max(d))
    return list(set(labels))[np.argmin(D)], np.max(softmax([-1 * np.array(D)]))


def hierarchical_grouping(
    distance_matrix,
    n_clusters: int,
    method: str = "complete",
    use_sklearn: bool = True,
    return_groups: bool = True,
    return_labels: bool = False,
):
    """hierarchical agglomerative grouping into n_clusters

    args
    ----
    distance_matrix : np.ndarray
        matrix of pairwise distances (works with any distance)

    n_clusters : int
        objective number of groups

    method : str
        linkage: single or complete
        NOTE: single linkage is not supported by sklearn

    use_sklearn : Boolean
        whether sklearn hierarchical clustering is used (same results, though a bit faster)

    return_labels : Boolean
        returns a single list of labels assigned to every series (ordered according to their fields)

    return_groups : Boolean
        returns a list of groups (ordered by their labels) of series' fields indices.

    TODO: add support for condensed distance matrix
    """
    assert return_labels != return_groups, "one type of output (groups/labels) must be chosen"

    if distance_matrix.ndim == 1:
        distance_matrix = squareform(distance_matrix)

    if use_sklearn:
        np.fill_diagonal(distance_matrix, sys.maxsize)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage=method
        ).fit(distance_matrix)
        labels = clustering.labels_
        if return_labels:
            return labels
        elif return_groups:
            group_list = get_indices_sparse(labels)
            return sorted(group_list, key=min)

    else:
        n_groups = distance_matrix.shape[1]  # elementary grouping initially
        G = [[i] for i in range(n_groups)]  # list of groups
        np.fill_diagonal(distance_matrix, sys.maxsize)
        D = np.copy(distance_matrix)

        while n_groups != n_clusters:
            closest_elems = np.argwhere(D == np.min(D))[0]

            G = update_list(G, closest_elems)  # new group comes first
            distance_new_bit = linkage(G[0], G[1:], distance_matrix, method=method)
            D = update_matrix(
                D, closest_elems, distance_new_bit, pad_value=sys.maxsize
            )  # update distance matrix according to linkage

            n_groups -= 1

        if return_labels:
            return convert_to_labels(G)
        elif return_groups:
            return sorted(G, key=min)


def convert_to_labels(groups: list):
    """converts a list of groups of fields' indices into a list of
    labels (ordered according to fields)
    """
    assert isinstance(groups, list)
    labels = np.zeros(len(sum(groups, [])), dtype=int)
    for k, g in enumerate(groups):
        labels[g] = k
    return labels


def linkage(A: list, B: list, dist_matrix: np.ndarray, method="complete"):
    """
    Args
    ----
    A : list
        a list of 2 or more indices (1 group)

    B : list
        a list of list of (n-1) (elementary) groups of indices

    dist_matrix : np.ndarray
        [n x n] matrix of pairwise min/max distances between
        clusters elements

    method : str
        linkage: single or complete

    Returns
    -------
    new array for distance matrix update:
        - single: min{d(a,b) | a \in A, b \in B} or
        - complete: max{d(a,b) | a \in A, B \in B}
    where d is the chosen metric (log-euclidean by default) and A, B are arrays of
    indices relative to spd matrices
    """
    A = [A] * len(B)
    links_ = []
    for a, b in zip(A, B):
        temp = []
        for c in list(itertools.product(a, b)):
            temp.append(dist_matrix[c[0], c[1]])
        if method == "single":
            links_.append(min(temp))
        elif method == "complete":
            links_.append(max(temp))
    return np.array(links_)


def update_matrix(M, closest_elem, new_arr, pad_value=0):
    """update adjacency/distance (square) matrix after one iteration of grouping

    Args
    ----
    closest_elem : ndarray
        indices of 2 closest components (and of row/column to delete from M)

    new_arr : ndarray
        array to add to M (first row and first column by default)

    pad_value : int, optional
        padding value at (0,0)

    NOTE:
    - new_arr size is [1, M.shape[0]-2]
    - output's size is [M.shape[0]-1, M.shape[1]-1] with pad_value at (0,0)
    """
    M_del_row = np.delete(M, closest_elem, axis=0)
    M_del_rc = np.delete(M_del_row, closest_elem, axis=1)
    M_add_col = np.hstack((new_arr.reshape(-1, 1), M_del_rc))
    return np.vstack((np.hstack((pad_value, new_arr)), M_add_col))


def update_list(groups: list, closest_elem):
    """update list of groups: merge 2 closest groups, puts it at the start and update remaining list of elements"""
    l = set(range(len(groups))) - set([closest_elem[0]]) - set([closest_elem[1]])
    g = [groups[closest_elem[0]] + groups[closest_elem[1]]] + [groups[k] for k in l]
    return g


def distance_to_centroids(X: np.ndarray, centroids: np.ndarray, logX=False, logC=False):
    """log-euclidean distances between spd matrices

    args
    ----
    logX : Boolean
            whether X are log-covariance matrices
    logC : Boolean
            whether centroids are log-spd matrices
    """
    X_ = X if logX else [logm(x) for x in X]
    centroids_ = centroids if logC else [logm(C) for C in centroids]
    D = np.zeros((len(centroids), len(X)))

    for i, c in enumerate(centroids_):
        for j, x in enumerate(X_):
            D[i, j] = logEuclidean_dist(x, c, logX=True, logY=True)
    return D


def update_centroids(X: np.ndarray, labels: np.ndarray, log=False):
    """computes the centroid (log-)covariance matrices"""
    centroids = []
    labels_idx = get_indices_sparse(labels)
    for idx in labels_idx:
        centroids.append(logEuclidean_mean(X[idx], log=log))
    return np.array(centroids)


def update_log_centroids(X: np.ndarray, labels: np.ndarray):
    """computes the centroid of log-spd matrices in the tangent plane (log-centroid)"""
    centroids = []
    labels_idx = get_indices_sparse(labels)
    for idx in labels_idx:
        centroids.append((1 / len(X[idx])) * np.add.reduce(X[idx]))
    return np.array(centroids)


def get_indices_sparse(X):
    """https://stackoverflow.com/questions/33281957/faster-alternative-to-numpy-where"""
    M = compute_sparse_matrix(X)
    return [np.unravel_index(row.data, M.shape)[1] for row in M]


def compute_sparse_matrix(X):
    cols = np.arange(X.size)
    return csr_matrix((cols, (X.ravel(), cols)), shape=(X.max() + 1, X.size))


def eigh(X):
    """spectral decomposition of X (spd)"""
    ev, U = np.linalg.eigh(X)
    idx = ev.argsort()[::-1]
    ev = ev[idx]
    U = U[:, idx]
    return ev, U


def logm(X):
    """returns log matrix of X"""
    ev, U = eigh(X)
    return np.linalg.multi_dot([U, np.diag(np.log(ev)), U.T])


def expm(X):
    """may suffer from overlfow for large eigenvalues..."""
    ev, U = eigh(X)
    return np.linalg.multi_dot([U, np.diag(np.exp(ev)), U.T])


def logEuclidean_dist(X, Y, logX=False, logY=False):
    """log-euclidean distance between two spd matrices X and Y (logX=True if log is precomputed)"""
    M = X if logX else logm(X)
    N = Y if logY else logm(Y)
    return np.linalg.norm(M - N, "fro") ** 2


def logEuclidean_mean(X, log=False):
    """log-euclidean mean of covariance matrices in X

    X : np.ndarray
        array of (log) spd matrices

    log: Boolean
            whether X is an array of log covariance matrices
    """
    assert len(X) > 0, "cannot average an empty array"
    X_ = X if log else [logm(x) for x in X]
    return expm((1 / len(X_)) * np.add.reduce(X_))


def distance_matrix(X, condensed=False, log=False):
    """pairwise logeuclidean distances

    args
    ----
    log : boolean
            whether X are log-spd matrices
    """
    L = []
    k, l = np.triu_indices(len(X), 1)
    for i, j in zip(k, l):
        L.append(logEuclidean_dist(X[i], X[j], logX=log, logY=log))
    if condensed:
        return np.array(L)
    else:
        return squareform(L)


# measures


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


def embed_np(x, window_size: int):
    """computes the trajectory matrix of x (L-embedding)

    Args
    ----
    x : ndarray
        input series
        if it is given k time series of length N, we assumed they are in a [k x N] matrix

    window_size : int
        window size
    """
    nd = x.ndim
    if nd == 1:
        N = len(x)
    elif nd == 2:
        N = x.shape[1]
    else:
        raise ValueError("does not support dim>2")
    K = N - window_size + 1
    if nd == 1:
        return hankel(x, np.zeros(window_size))[:K]
    else:
        l = []
        for x_ in x:
            l.append(hankel(x_, np.zeros(window_size))[:K])
        return np.hstack(l)


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
