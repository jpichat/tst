# TST

### Overview
The Time Series Toolbox, TST (wip) currently delivers functionalities for:
- transforms
   - [decomposition](tst/transform/decompose.py)
   - [missing values imputation](tst/transform/impute.py)
- clustering
   - [covariance-kmeans](tst/cluster/covariance_kmeans.py)
   - [covariance-hierarchical](tst/cluster/covariance_hierarchical.py)
- various simple utilities in `tst/utils` (in no specific order)

### Docs
- notebooks
   - [SSA](docs/notebooks/ssa.ipynb)
- notes
   - [ARIMA](docs/notes/arima.pdf)
   - [RNN](docs/notes/rnn.pdf)
   - [Singular Spectrum Analysis](docs/notes/ssa.pdf)

### To Do 
- add/integrate baseline models
- add/integrate simple transforms (feature engineering, selection)
- add support for end-to-end training (pipelines)
- add simple visualisations
