import time
import typing
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import softmax

from tst.utils.utils import covariance, trim_nans, svd_smoother, embed, windowed_name
from tst.utils.clustering import (
    distance_to_centroids,
    update_log_centroids,
    logEuclidean_dist,
    update_centroids,
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
    centroids_stable = update_centroids(arr, labels_stable, log=True)

    return labels_stable, centroids_stable
