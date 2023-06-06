import time
import logging
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from lib.utils import weighted_correlation, dot_avg_ma, dot_avg_at, embed, windowed_name
from lib.cluster.kmeans import hierarchical_grouping


class GroupingMethods:
    WCORR = "wcorr"
    EIGEN = "eigen"


class SSA(TransformerMixin, BaseEstimator):
    """
    Decomposition of a 1d series using SSA [1] into n additive time components.
    The decomposition works with the whole data in memory.

    .. note:: the input series is assumed to be centred and scaled to unit variance.

    References
    ----------
    .. [1] (Golyandina et al., 2001) Analysis of Time Series Structure: SSA and related techniques
    .. [2] (Golyandina et al. 2019) Particularities and commonalities of singular spectrum analysis
    as a method of time series analysis and signal processing https://arxiv.org/pdf/1907.02579.pdf
    """

    SUFFIX = "_SSA"

    def __init__(
        self,
        field: str,
        embedding_dimension: int = None,
        n_components: int = None,
        n_eigenvectors: int = None,
        var_threshold: int = None,
        grouping_method: str = GroupingMethods.WCORR,
        force_isolate_trend: bool = True,
        verbose=False,
    ):
        """
        Args
        ----
        field : str
            series to decompose

        embedding_dimension : int
            covariance dimension

        n_components : int, optional
            number of time components to output

        n_eigenvectors : int, optional (defaulted to k that explain
            more than 99.5% of the variance of the signal) number of leading eigenvectors to consider for projections

        var_threshold : int (optional, default: 99 [0-100])
            lower bound on the proportion of total variance explained by the leading eigenvectors

        grouping_method : str (optional, default: None)
            . None : no grouping
            . wcorr : weighted correlations between components
            . eigen : Euclidean distance between eigenvalues

        force_isolate_trend : bool
            keeps first component (trend) as elementary and prevents it from being grouped with other elementary
            components (if grouping_method is not None)
            .. note:: in theory, the trend should already be little correlated to other components but it may happen
            that it unwillingly gets grouped with other components due to some parameters combination
        """
        assert isinstance(field, str), "expected a str but received {}".format(type(field))
        assert embedding_dimension is not None, "missing embedding dimension!"

        if n_components is not None:
            assert n_components > 0, "number of components should be >0"
            assert (
                n_components <= embedding_dimension
            ), "The requested number of components exceeds the embedding dimension"

        if n_eigenvectors is not None:
            assert n_eigenvectors > 0, "number of eigenvectors should be >0"
            assert (
                n_eigenvectors <= embedding_dimension
            ), "The requested number of eigenvectors exceeds the embedding dimension"

        if n_eigenvectors is not None and n_components is not None:
            assert (
                n_components <= n_eigenvectors
            ), "The number of eigenvectors should be smaller than the number of components"

        if var_threshold is not None and not isinstance(var_threshold, str):
            assert 0 < var_threshold < 100, "variance threshold should be between 0 and 100"

        self.field = field
        self.embedding_dimension = embedding_dimension
        self.n_components = n_components
        self.n_eigenvectors = n_eigenvectors
        self.var_threshold = var_threshold
        self.grouping_method = grouping_method
        self.force_isolate_trend = force_isolate_trend
        self.verbose = verbose
        # attributes set during fit
        self.out_dim = None
        self.lags = None
        self.e_vals = None
        self.e_vecs = None
        self.proj_mat = None
        self.groups = None

    def _set_subspace_dimension(self):
        """sets the max number of eigenvectors to use for decomposition (k)

        .. note:: auto var_thresh relies on https://arxiv.org/pdf/1305.5870.pdf
        """

        VAR_THRESH = 99.5
        self.grouping = True
        C = 100.0 * np.cumsum(self.e_vals) / np.sum(self.e_vals)  # cumulated contribution

        if self.n_components == 1 and self.grouping_method is not None:
            logging.info(">>> Grouping disabled ('n_components=1').")
            self.grouping_method = None

        if self.n_eigenvectors is None:

            if self.var_threshold is None:
                # force variance threshold to VAR_THRESH=99.5%
                self.out_dim = np.argwhere(C >= VAR_THRESH)[0][0] + 1
            elif isinstance(self.var_threshold, (int, float)):
                self.out_dim = np.argwhere(C >= self.var_threshold)[0][0] + 1
            elif self.var_threshold == "auto":
                s_vals = np.sqrt(self.e_vals)
                self.out_dim = len(s_vals[s_vals > 2.858 * np.median(s_vals)])
            else:
                raise ValueError("unknown thresholding")

            logging.info(
                f">>> Suggestion: use {self.out_dim}/{self.embedding_dimension} "
                f"eigenvector(s) ({np.round(C[self.out_dim - 1], 2)}%)"
            )

            if self.n_components is None:
                self.n_components = self.out_dim
                logging.info(f">>> Forcing decomposition into {self.out_dim} component(s)")
            else:
                if self.out_dim < self.n_components:
                    self.grouping = False
                    logging.info(
                        ">>> No grouping needed: forced using more eigenvectors than necessary: "
                        f"{self.out_dim}->{self.n_components}/{self.embedding_dimension} "
                        f"({np.round(C[self.out_dim - 1], 2)}->{np.round(C[self.n_components - 1], 2)}%)"
                    )
                    self.out_dim = self.n_components
                elif self.out_dim == self.n_components:
                    self.grouping = False
                    logging.info(">>> No grouping needed: elementary decomposition")

            if self.grouping_method is None:
                self.grouping = False
                self.out_dim = self.n_components
                logging.info(
                    f">>> No grouping requested: subspace dimension is forced to {self.out_dim}/{self.embedding_dimension}"
                )

        else:
            # use the requested number of eigenvectors
            if self.var_threshold is not None:
                logging.info(">>> Variance threshold was ignored.")

            if self.n_components is None:
                self.grouping = False
                self.n_components = self.n_eigenvectors
                logging.info(
                    ">>> No grouping: elementary decomposition into "
                    f"{self.n_eigenvectors}/{self.embedding_dimension} components "
                    f"({ np.round(C[self.n_eigenvectors - 1], 2)}%)"
                )
            else:
                if self.n_components == self.n_eigenvectors:
                    self.grouping = False

            if self.grouping_method is None:
                # ignoring number of eigenvectors
                self.grouping = False
                logging.info(
                    f">>> Requested no grouping: using {self.n_eigenvectors}"
                    f"->{ self.n_components}/{self.embedding_dimension} "
                    f"eigenvectors for elementary decomposition ({np.round(C[self.n_components - 1], 2)}%)"
                )
                self.out_dim = self.n_components
            else:
                logging.info(
                    f">>> Grouping {self.n_eigenvectors} elementary components "
                    f"({np.round(C[self.n_eigenvectors - 1], 2)}%) into {self.n_components}"
                )
                self.out_dim = self.n_eigenvectors

    @staticmethod
    def _optimal_svht_coef_sigma_known(cls, beta):
        w = 8 * beta / (beta + 1.0 + np.sqrt(beta ** 2 + 14.0 * beta + 1.0))
        lambda_star = np.sqrt(2 * (beta + 1) + w)
        return lambda_star

    @staticmethod
    def _median_marcenko_pastur(cls, beta):
        raise NotImplementedError

    @staticmethod
    def _optimal_SVHT_coef_sigma_unknown(cls, beta):
        coef = cls._optimal_svht_coef_sigma_known(beta)
        mp_median = []
        for i in range(len(beta)):
            mp_median.append(cls._median_marcenko_pastur(beta[i]))
        omega = coef / np.sqrt(mp_median)

    def _covariance(self, chunk: pd.DataFrame):
        """lag-covariance matrix"""
        chunk_values = chunk[self.lags].dropna().values
        return np.dot(chunk_values.T, chunk_values)

    def fit(self, chunk: pd.DataFrame):
        start = time.time()

        # window
        chunk2d = embed(chunk, self.field, self.embedding_dimension, keep_dims=False)
        self.lags = [
            windowed_name(self.field, t, "") for t in range(-self.embedding_dimension + 1, 1)
        ]

        # decompose lag-covariance matrix
        e_vals, e_vecs = np.linalg.eigh(self._covariance(chunk2d))
        idx = e_vals.argsort()[::-1]
        self.e_vals = e_vals[idx]
        self.e_vecs = e_vecs[:, idx]
        self._set_subspace_dimension()  # sets self.out_dim
        del e_vals, e_vecs, idx

        logging.info(f">>> SSA decomposition of '{self.field}' into {self.out_dim} components")
        logging.info(f">>>    | embedding dimension: {self.embedding_dimension}")

        # projection matrices
        self.proj_mat = [
            np.einsum("i,j->ij", self.e_vecs[:, i], self.e_vecs[:, i]) for i in range(self.out_dim)
        ]

        if self.grouping and self.out_dim > 1:
            # TODO: implement progress reporting here?
            self.groups = self._group(chunk)  # FIXME: grouping should move to transform
        else:
            self.groups = [[i] for i in range(self.out_dim)]

        logging.info(f">>> SSA done! ({np.round(time.time()-start, 3)}s)")

        return self

    def _group(self, chunk):
        """grouping components"""
        decomposition = None  # None 3 if grouping method is "eigen"
        if self.grouping_method == GroupingMethods.WCORR:
            decomposition = self.transform(chunk)[self.output_fields]

        # grouping
        dist_mat = self.distance_matrix(series=decomposition)
        if self.force_isolate_trend:
            dist_mat = dist_mat[1:, 1:]
            groups = hierarchical_grouping(dist_mat, self.n_components - 1)
            groups = [g + 1 for g in groups]  # shifts indices
            groups.insert(0, np.array([0]))
        else:
            groups = hierarchical_grouping(dist_mat, self.n_components)

        self.proj_mat = [np.add.reduce([self.proj_mat[i] for i in g]) for g in groups]

        groups = [g.tolist() for g in groups]

        extra_log = f" (first {len(groups[:5])})" if len(groups) > 5 else ""
        logging.info(f">>> Grouping done! {str(groups)}" + extra_log)

        return groups

    def transform(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """follows sklearn naming to avoid confusion with mlcore apply, which works by chunk"""
        try:
            if self.embedding_dimension == 1:
                return chunk

            assert len(chunk) > 2 * (
                self.embedding_dimension - 1
            ), f"There should be >{2*(self.embedding_dimension-1)} observations for the chosen embedding dimension ({self.embedding_dimension})"

            # missing values
            mask = chunk[self.field].isnull()
            chunk2d = embed(chunk, self.field, self.embedding_dimension, keep_dims=False)
            M = chunk2d.fillna(0).values  # missing values are filled with 0s
            R = pd.DataFrame(dtype=np.float)

            for i, P in enumerate(self.proj_mat):
                out = np.empty(0)
                out = np.append(out, dot_avg_at(M[: self.embedding_dimension], P))
                for j in range(M.shape[0] - self.embedding_dimension + 1):
                    out = np.append(out, dot_avg_ma(M[j : j + self.embedding_dimension], P))
                out = np.append(out, dot_avg_at(M[-self.embedding_dimension :], P, which="lower"))
                R[self.output_fields[i]] = out

            # add left-over columns from input to result
            R[chunk.columns] = chunk[chunk.columns]

            # put back mv from mask
            R[self.field].where(~mask, other=np.nan, inplace=True)

            return R

        except AttributeError:
            raise NotFittedError("transform should be fitted first")

    @property
    def output_fields(self):
        try:
            d_ = len(self.groups) + 1
        except AttributeError:
            d_ = self.out_dim + 1
        return [self.field + self.SUFFIX + f"{i}" for i in range(1, d_)]

    def distance_matrix(self, series: pd.DataFrame = None, ignore_original=True):
        """returns weighted-correlation distance matrix

        Args
        ----
        series : pd.DataFrame
            series to be compared (along columns)

        ignore_original : bool
            whether weighted correlations between original series and its components are computed
        """
        if series is not None:
            # assumes computation of wcorr
            assert len(series.columns) >= 2, "needs at least 2 series to compute pairwise distances"
            if ignore_original:
                x = series[self.output_fields].values.T
            else:
                x = series[[self.field] + self.output_fields].values.T
            return 1 - weighted_correlation(x, self.embedding_dimension)
        else:
            # WARN: while the wcorr-based distance matrix may include pairwise comparisons with
            # the original series, eigen-distance matrix doesn't...
            ev_ = self.e_vals[: self.out_dim]
            if hasattr(self, "groups"):
                ev_ = [np.mean(ev_[g]) for g in self.groups]
            return squareform(pdist(ev_.reshape(-1, 1)))

    @property
    def explained_variance(self):
        return (100.0 * np.cumsum(self.e_vals) / np.sum(self.e_vals))[self.out_dim - 1]

    @property
    def eigenbasis(self):
        """orthonormal basis of eigenvectors along columns"""
        try:
            return self.e_vecs[:, : self.out_dim]
        except AttributeError:
            raise NotFittedError("transform should be fitted first")

    @property
    def eigenvalues(self):
        """ordered eigenvalues of lag-covariance matrix"""
        try:
            return self.e_vals[: self.out_dim]
        except AttributeError:
            raise NotFittedError("transform should be fitted first")

    def component_fields(self, k=None):
        """returns all(k=None)/some components fields (after grouping)

        args:
        -----
        k : int, list/range/np.ndarray (optional, default: None)
            components output fields
        """
        if k is None:
            return [self.field + self.SUFFIX + f"{k}" for k in range(1, self.n_components + 1)]

        elif isinstance(k, int):
            assert 1 <= k <= self.n_components, "requested component does not exist"
            return [self.field + self.SUFFIX + f"{k}"]

        elif isinstance(k, (list, range, np.ndarray)):
            k = np.array(k)
            assert all(
                (k >= 1) & (k <= self.n_components)
            ), "some requested components do not exist"
            return [self.field + self.SUFFIX + f"{k_}" for k_ in k]

        else:
            raise TypeError("cannot return the requested ssa field(s)")

    def residual(self, chunk: pd.DataFrame, suffix=None) -> pd.DataFrame:
        """ssa-based noise estimation in 1d series

        Args
        ----
        chunk : DataFrame
            input 1d series

        .. note:: use with care! if too little variance is explained along the principal directions
        (n_eigenvectors<<L or small var_threshold), then the residual is likely to still contain signal!
        """
        s = "" if suffix is None else suffix
        try:
            dd = self.transform(chunk)[self.output_fields]
        except AttributeError:
            raise NotFittedError("transform should be fitted first")
        residual = chunk[self.field] - np.sum(dd, axis=1)
        return residual.to_frame(self.field + s)

    def serialize(self):
        return dict(
            field=self.field,
            embedding_dimension=self.embedding_dimension,
            n_components=self.n_components,
            n_eigenvectors=self.n_eigenvectors,
            var_threshold=self.var_threshold,
            grouping_method=self.grouping_method,
            projection_matrices=self.proj_mat,
            groups=self.groups,
            lags=self.lags,
            force_isolate_trend=self.force_isolate_trend,
        )

    @classmethod
    def deserialize(cls, data):
        v = cls(data["field"], embedding_dimension=data["embedding_dimension"])
        v.n_components = data["n_components"]
        v.n_eigenvectors = data["n_eigenvectors"]
        v.var_threshold = data["var_threshold"]
        v.grouping_method = data["grouping_method"]
        v.proj_mat = data["projection_matrices"]
        v.groups = data["groups"]
        v.lags = data["lags"]
        v.force_isolate_trend = data["force_isolate_trend"]
        return v
