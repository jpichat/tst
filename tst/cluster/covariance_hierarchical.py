import time
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from tst.utils.utils import covariance, trim_nans, embed, windowed_name, svd_smoother
from tst.utils.clustering import (
    hierarchical_grouping,
    distance_matrix,
    assign_cluster,
    logm,
)


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
