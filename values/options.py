import typing


class DatasetOptions:
    def __init__(
        self,
        include: typing.List[str] = None,
        exclude: typing.List[str] = None,
        targets: typing.List[str] = None,
        na_values: typing.Mapping[str, typing.List[str]] = None,
        types: typing.Mapping[str, str] = None,
        formulas: typing.Mapping[str, str] = None,
        inclusive_modeling: bool = False,
    ):
        self.include = include or None
        self.exclude = exclude or None
        self.targets = targets or None
        self.na_values = na_values or None
        self.types = types or None
        self.formulas = formulas or None
        self.inclusive_modeling = inclusive_modeling

    @staticmethod
    def from_client(dataset_options: typing.Optional[tt.DatasetOptions]):
        """
        Create :py:class:`DatasetOptions` object from client arguments.
        """
        if dataset_options is None:
            return None

        fields = []
        ignore = []
        targets = []
        na_values = {}
        types = {}
        formulas = {}
        inclusive_modeling = dataset_options.get("inclusiveModeling", False)

        for field_settings in dataset_options["customFieldSettings"]:
            field = field_settings["name"]
            role = field_settings.get("role")
            field_type = field_settings.get("type")

            if (
                field_settings.get("transformOptions")
                and "formula" in field_settings["transformOptions"]
            ):
                formulas[field] = field_settings["transformOptions"]["formula"]
                continue  # formula fields cannot be targets or have type information nor missing values

            if role == FieldRoles.ignore:
                ignore.append(field)
            elif role == FieldRoles.target:
                targets.append(field)
            elif field_type != FieldTypes.numeric and field_type != FieldTypes.category:
                ignore.append(field)
                mlcore.logger.warning(
                    "auto-ignoring field {} because of type {}".format(field, field_type)
                )
            else:
                fields.append(field)

            if "missingValues" in field_settings:
                na_values[field] = field_settings["missingValues"]
            if field_type:
                types[field] = field_type
        return DatasetOptions(
            fields, ignore, targets, na_values, types, formulas, inclusive_modeling
        )

    def __repr__(self):
        return "DatasetOptions(include={}, exclude={}, targets={}, na_values={}, types={}, inclusive_modeling={})".format(
            self.include,
            self.exclude,
            self.targets,
            self.na_values,
            self.types,
            self.inclusive_modeling,
        )

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__


class DatetimeOptions:
    def __init__(self, field: str = None, format: str = None, resolution: str = None):
        self.field = field
        self.format = format
        self.resolution = resolution

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"DatetimeOptions(field={self.field}, format={self.format}, resolution={self.resolution})"

    @staticmethod
    def from_client(datetime_options: typing.Optional[tt.DatetimeOptions]):
        if datetime_options is None:
            return None
        return DatetimeOptions(
            datetime_options["field"],
            datetime_options["format"],
            datetime_options.get("resolution"),
        )


class ImputeMethods:
    class Simple:
        ROLLING = "rolling"
        CONSTANT = "constant"

    class Complex:
        SSA = "ssa"
        GP = "gp"
        SVT = "svt"

    @classmethod
    def simple_types(cls):
        return {cls.Simple.ROLLING, cls.Simple.CONSTANT}

    @classmethod
    def complex_types(cls):
        return {cls.Complex.SSA, cls.Complex.GP, cls.Complex.SVT}

    @classmethod
    def all(cls):
        return cls.simple_types().union(cls.complex_types())


class ImputeParams:
    """class for all imputer parameters"""

    # equivalence FEBE/MLCore
    PARAMS_MAP = {
        "fillValue": "fill_with",
        "strategy": "strategy",
        "windowSize": "window_size",
        "embeddingDimension": "embedding_dimension",
        "numberOfComponents": "n_components",
        "varianceThreshold": "var_threshold",
        "useKComponents": "use_k_components",
        "maxIterations": "max_iterations",
        "kernel": "kernel",
        "noiseLevel": "noise_level",
        "maxEvaluations": "max_evals",
        "solver": "solver",
    }

    # every method params set
    # NOTE: not using "backward" intentionally, to avoid leaking information from the future
    METHOD_PARAMS_MAP = {
        "constant": ("fill_with", "strategy"),
        "rolling": ("window_size", "strategy"),
        "ssa": (
            "n_components",
            "embedding_dimension",
            "var_threshold",
            "use_k_components",
            "max_iterations",
        ),
        "gp": ("kernel", "noise_level", "max_evals"),
        "svt": (
            "embedding_dimension",
            "solver",
            "max_iterations",
        ),
    }

    def __init__(
        self,
        fill_with: typing.Optional[typing.Union[float, str]] = None,
        strategy: typing.Optional[str] = None,
        window_size: typing.Optional[int] = None,
        embedding_dimension: typing.Optional[int] = None,
        n_components: typing.Optional[int] = None,
        var_threshold: typing.Optional[float] = None,
        use_k_components: typing.Optional[int] = None,
        max_iterations: typing.Optional[int] = None,
        kernel: typing.Optional[str] = None,
        noise_level: typing.Optional[float] = None,
        max_evals: typing.Optional[float] = None,
        solver: typing.Optional[str] = None,
    ):
        self.fill_with = fill_with
        self.strategy = strategy
        self.window_size = window_size
        self.embedding_dimension = embedding_dimension
        self.n_components = n_components
        self.var_threshold = var_threshold
        self.use_k_components = use_k_components
        self.max_iterations = max_iterations
        self.kernel = kernel
        self.noise_level = noise_level
        self.max_evals = max_evals  # noqa
        self.solver = solver

    def get(self, method: str) -> dict:
        """get the set of parameters for a given imputation method"""
        return {
            k: getattr(self, k)
            for k in ImputeParams().__dict__.keys()
            if k in self.METHOD_PARAMS_MAP[method]
        }

    @classmethod
    def from_client(cls, params: typing.List[tt.Param]) -> "ImputeParams":
        """translates FEBE Param (list of dicts) into MLCore imputers params"""

        impute_params = ImputeParams()

        for p in params:
            impute_params.__setattr__(cls.PARAMS_MAP[p["name"]], p.get("value") or p.get("default"))

        return impute_params


class ImputeOptions:
    ASSESS_ALL_TARGETS = "all"

    class Method(typing.NamedTuple):
        name: ImputeMethods
        params: dict

    class AssessBenefit:
        def __init__(self, enabled: bool = False, target: str = None):
            self.enabled = enabled
            self.target = target  # a given field or a str: "all"

        def __bool__(self):
            if self.enabled:
                assert self.target is not None, "assessment requires a target"
                assert isinstance(self.target, str), type(str)
                return True
            else:
                return False

        def __repr__(self):
            if self.enabled:
                return f"AssessBenefit(enabled={self.enabled}, target={self.target})"
            else:
                return f"AssessBenefit(enabled={self.enabled})"

    def __init__(
        self,
        auto: typing.Optional[bool] = False,
        use_complex_methods: typing.Optional[bool] = False,
        method: typing.Optional[Method] = None,
        assess_benefit: typing.Optional[AssessBenefit] = None,
    ):
        """
        Args
        ----
        auto : bool
            whether to go for auto-imputation

        use_complex_methods : bool
            whether to use complex auto-imputation

        method: Method
            name and params of the chosen imputation strategy

        assess_benefit: AssessBenefit
            check whether imputation improves modeling in some sense
        """
        self.auto = auto
        self.use_complex_methods = use_complex_methods
        self.method = method
        self.assess_benefit = assess_benefit

    @classmethod
    def from_client(cls, impute_options: tt.ImputeOptions) -> "ImputeOptions":
        auto = impute_options.get("auto", False)
        use_complex_methods = impute_options.get("useComplexMethods", False)
        assess_benefit = impute_options.get("assessBenefit", False)
        method = impute_options.get("method")
        assess_benefit = impute_options.get("assessBenefit")
        if method is not None:
            method = cls.Method(
                name=method["name"],
                # convert params to MLCore transform params
                params=ImputeParams.from_client(method["params"]).get(method["name"]),
            )
        return ImputeOptions(
            auto=auto,
            use_complex_methods=use_complex_methods,
            method=method,
            assess_benefit=cls.AssessBenefit(
                enabled=assess_benefit["enabled"],
                target=assess_benefit.get("target"),
            ),
        )

    def __repr__(self):
        if self.single_auto:
            return f"ImputeOptions(mode={self.mode}, assess_benefit={self.assess_benefit})"
        else:
            return f"ImputeOptions(method={self.method}, assess_benefit={self.assess_benefit})"
