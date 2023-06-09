import time
import typing
import logging
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import softmax

from tst.utils.utils import (
    covariance,
    trim_nans,
    diag_avg,
    embed,
    windowed_name,
)
from tst.utils.clustering import (
    hierarchical_grouping,
    distance_to_centroids,
    distance_matrix,
    update_log_centroids,
    logEuclidean_dist,
    update_centroids,
    assign_cluster,
    logm,
)


class CovarianceKMeans(BaseEstimator):
    """divisive clustering in Riemannian geometry of time series covariance matrices

    assumptions
    -----------
    input to clustering is a datasource of time series (per column) all of which:
    - (i) are temporally aligned: shorter ones are pre/postpended with NaNs, so they all refer to the same time scale
    - (ii) are filled (actual missing values are not NaNs anymore, they're either 0s or something better)
    """

    def __init__(
        self,
        fields: typing.List[str],
        n_clusters: int,
        embedding_dimension: typing.Union[typing.List[int], int] = 12,
        n_components: int = None,
        n_repetitions: int = 5,
        return_trends: bool = False,
        verbose: bool = False,
    ):
        """
        Args
        ----
        n_clusters : int
            objective number of labels to output

        embedding_dimension : int or list
            dimension of covariance matrices (if an integer is given, all series'
            covariance matrices have the same dimensions, otherwise, each matrix has a given size)

        n_components : int (default: None)
            number of principal components to consider for smoothing: smoothed series are
            used in place of original series for covariance grouping

        return_trends : boolean
            whether smoothed series are stored (None if n_components is None)

        n_repetitions : int
            number of runs from which the most stable labels are picked
        """
        assert len(fields) > 1, "needs more than 1 series to perform clustering"
        assert len(fields) > n_clusters, "needs more series than groups to perform clustering"

        self.fields = fields
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.return_trends = return_trends
        self.n_repetitions = n_repetitions
        self.verbose = verbose
        self.labels = None
        self.centroid = None
        self.trends = None

        if isinstance(embedding_dimension, int):
            self.embedding_dimension = [embedding_dimension] * len(self.fields)
        elif isinstance(embedding_dimension, list):
            assert len(embedding_dimension) == len(self.fields)
            self.embedding_dimension = embedding_dimension

    def fit(self, df: pd.DataFrame):
        df = df[self.fields]
        # prepare log-covariance matrices
        start = time.time()
        log_cov, self.trends = self.prepare_covariance(df, log=True)
        if self.verbose:
            t1 = np.round(time.time() - start, 3)
            logging.info(f">>> log-covariances done... in {t1}s")

        # grouping
        if self.verbose:
            start = time.time()
        self.labels, self.centroids = log_kmeans(
            log_cov, self.n_clusters, n_repetitions=self.n_repetitions
        )
        if self.verbose:
            t2 = np.round(time.time() - start, 3)
            logging.info(
                f">>> fit done... ({self.n_repetitions} times) in {t2}s (total: {t1 + t2}s)"
            )

    def predict(self, df: pd.DataFrame):

        assert len(self.fields) <= len(
            self.embedding_dimension
        ), f"size of training set ({len(self.embedding_dimension)}) < number of labels to be predicted ({len(self.fields)})..."

        df = df[self.fields]
        log_cov = self.prepare_covariance(df, log=True)[0]

        # self.centroids belong to the domain of spd matrices
        D = distance_to_centroids(log_cov, self.centroids, logX=True)

        # confidence into the assignment of X to the closest centroid's label
        score = np.max(softmax(-D.T), axis=1)

        return dict(zip(self.fields, np.argmin(D, axis=0))), dict(zip(self.fields, score))

    def prepare_covariance(self, df: pd.DataFrame, log=False):
        """prepare covariance matrices of raw/smoothed time series

        Args
        ----
        df: pd.DataFrame
            input series

        log : bool
            whether log-covariance matrices are computed
        """
        trends = None
        if self.return_trends and self.n_components is not None:
            trends = []

        cov = []

        assert isinstance(
            self.embedding_dimension, list
        ), f"expecting embedding dimension to be a list of length {len(self.fields)}"

        for field, L in zip(self.fields, self.embedding_dimension):
            df_t = pd.DataFrame(trim_nans(df, field)[0], columns=[field])

            if self.n_components is not None:
                # working with smoothed series
                trend = svd_smoother(df_t, field, n_components=self.n_components)
            else:
                trend = df_t

            if self.return_trends and self.n_components is not None:
                trends.append(trend.values)

            T = embed(df, [field], n_lags=L, keep_dims=False).drop(columns=[field])
            C = covariance(T, [windowed_name(field, i) for i in range(L)])

            if log:
                cov.append(logm(C))
            else:
                cov.append(C)

        return np.array(cov), trends


class CovarianceHierarchical(BaseEstimator):
    """hierarchical agglomerative grouping of time series in Riemannian geometry using distances between their covariance matrices

    Notes
    -----
    - as opposed to CovarianceKMeans, "single/complete-linkage" hierarchical clustering does not necessitate the computation of centroids
        and only requires the full distance matrix, computed once, at the beginning.
    - increasing the number of clusters is equivalent to thresholding the resulting dendrogram at decreasing values
    """

    def __init__(
        self,
        fields: list,
        n_clusters: int,
        embedding_dimension,
        n_components=None,
        return_trends=False,
        use_sklearn=True,
        method: str = "complete",
        verbose=False,
    ):
        """
        Args
        ----
        fields : list
                    list of series' fields

        n_clusters : int
            objective number of clusters

        embedding_dimension : list
            covariance matrices dimensions

        n_components : int
            number of components to use for series smoothing

        see also: hierarchical_grouping doc
        """
        self.fields = fields
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.return_trends = return_trends
        self.method = method
        if self.method == "single":
            assert (
                not use_sklearn
            ), "single linkage is not supported by sklearn hierarchical clustering"
        self.use_sklearn = use_sklearn
        self.verbose = verbose
        self.log_cov_fit = None
        self.trends = None
        self.labels = None
        self.result = None

        if isinstance(embedding_dimension, int):
            self.embedding_dimension = [embedding_dimension] * len(self.fields)
        elif isinstance(embedding_dimension, list):
            assert len(embedding_dimension) == len(self.fields)
            self.embedding_dimension = embedding_dimension

    def fit(self, df: pd.DataFrame):
        df = df[self.fields]
        # prepare covariance matrices
        if self.verbose:
            start = time.time()
        self.log_cov_fit, self.trends = self.prepare_covariance(df, log=True)
        if self.verbose:
            t1 = np.round(time.time() - start, 3)
            logging.info(f">>>log-covariances done... in {t1}s")

        # grouping
        if self.verbose:
            start = time.time()
        D = distance_matrix(self.log_cov_fit, log=True)
        self.labels = hierarchical_grouping(
            D,
            self.n_clusters,
            method=self.method,
            use_sklearn=self.use_sklearn,
            return_labels=True,
            return_groups=False,
        )
        if self.verbose:
            t2 = np.round(time.time() - start, 3)
            logging.info(f">>> grouping done... in {t2}s (total: {t1 + t2}s)")

        self.result = dict(zip(self.fields, self.labels))

    def predict(self, df: pd.DataFrame, fields, embedding_dimension, verbose=False):
        """assigns new series to a group according to a linkage method

        NOTE: for now, all series have the same embedding dimension,
        but it'd be nice if it was parameter...
        """
        if isinstance(fields, str):
            fields = [fields]

        if isinstance(embedding_dimension, int):
            embedding_dimension = [embedding_dimension] * len(fields)
        elif isinstance(embedding_dimension, list):
            assert len(embedding_dimension) == len(fields)

        df = df[fields]
        if verbose:
            start = time.time()
        log_cov_predict = self.prepare_covariance(df, log=True)[0]
        if verbose:
            logging.info(f">>> log-covariances done... in {np.round(time.time() - start, 3)}s")

        labels = []
        scores = []
        for x in log_cov_predict:
            label_, score_ = assign_cluster(
                x, self.log_cov_fit, self.labels, method=self.method, log=True
            )
            labels.append(label_)
            scores.append(score_)

        return dict(zip(fields, labels)), dict(zip(fields, scores))

    def prepare_covariance(self, df: pd.DataFrame, log=False):
        """prepare covariance matrices of raw/smoothed time series

        Args
        ----
        df: pd.DataFrame
            input series

        log : bool
            whether log-covariance matrices are computed
        """
        trends = None
        if self.return_trends and self.n_components is not None:
            trends = []

        cov = []

        assert isinstance(
            self.embedding_dimension, list
        ), f"expecting embedding dimension to be a list of length {len(self.fields)}"

        for field, L in zip(self.fields, self.embedding_dimension):
            df_t = pd.DataFrame(trim_nans(df, field)[0], columns=[field])

            if self.n_components is not None:
                # working with smoothed series
                trend = svd_smoother(df_t, field, n_components=self.n_components)
            else:
                trend = df_t

            if self.return_trends and self.n_components is not None:
                trends.append(trend.values)

            T = embed(df, [field], n_lags=L, keep_dims=False).drop(columns=[field])
            C = covariance(T, [windowed_name(field, i) for i in range(L)])

            if log:
                cov.append(logm(C))
            else:
                cov.append(C)

        return np.array(cov), trends


def svd_smoother(df: pd.DataFrame, field: str, embedding_dimension=None, n_components=1, seed=42):
    """
    SSA-like trend estimation of a series using the first k components of the
    svd decomposition of its trajectory matrix
    """
    # large window for coarse de-trending
    # L=min(len(df)//2, 512) if embedding_dimension is None else embedding_dimension
    embed_dim = embedding_dimension or 14
    df_ = embed(df, [field], n_lags=embed_dim, keep_dims=False).drop(columns=[field])
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(df_)
    H = np.add.reduce(
        [
            np.linalg.multi_dot(
                [
                    df_,
                    svd.components_[i].reshape(-1, 1),
                    svd.components_[i].reshape(1, -1),
                ]
            )
            for i in range(n_components)
        ]
    )
    return pd.DataFrame(diag_avg(H), columns=[field])


def log_kmeans(arr: np.ndarray, n_clusters: int, n_repetitions=1, thresh=1e-5, seed=42):
    """groups log-spd matrices into n_clusters according to chosen metric

    Args
    ----
    arr : np.ndarray
        array of log-spd matrices (all have the same dimensions for now; future work: [2])

    n_clusters : int
        number of clusters

    References
    ----------
    .. [1] Arsigny 2007 - Geometric Means in a Novel Vector Space Structure on
    Symmetric Positive‐Definite Matrices
    .. [2] Lim 2018 - Geometric distance between positive definite matrices
    of different dimensions
    """
    assert len(arr) >= n_clusters

    np.random.seed(seed)
    seeds_ = np.random.randint(0, 100 * n_repetitions, n_repetitions)

    labels = []

    for s in seeds_:

        # n_clusters random initial centroids
        idx = np.random.RandomState(seed=s).permutation(len(arr))[:n_clusters]
        centroids = arr[idx]  # log-spd matrices
        delta = np.inf

        while delta > thresh:
            dist_ = distance_to_centroids(arr, centroids, logX=True, logC=True)
            labels_ = np.argmin(dist_, axis=0)
            # we intentionally stay in the tangent planes to avoid going back and forth with the exponential map
            new_centroids = update_log_centroids(arr, labels_)
            delta = np.sum(
                [
                    logEuclidean_dist(M, N, logX=True, logY=True)
                    for M, N in zip(centroids, new_centroids)
                ]
            )
            centroids = new_centroids

        labels.append(labels_)

    # majority voting
    # FIXME: it may happen than there are less clusters than expected...
    labels_stable = np.array([np.argmax(np.bincount(l)) for l in np.array(labels).T])
    centroids_stable = update_centroids(X, labels_stable, log=True)

    return labels_stable, centroids_stable
