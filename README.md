# TST

## Overview
The Time Series Toolbox (TST) is a work in progress that currently delivers functionalities for:
- transforms
   - [decomposition](tst/transform/decompose.py)
      - Singular Spectrum Analysis (SSA)
   - [missing values imputation](tst/transform/impute.py)
      - in-place simple value imputation
      - fwd/bwd rolling window
      - SSA
      - Gaussian processes (GP)
      - Singular value thresholding (SVT) 
- clustering
   - [covariance-kmeans](tst/cluster/covariance_kmeans.py)
   - [covariance-hierarchical](tst/cluster/covariance_hierarchical.py)
- various simple utilities in `tst/utils` (in no specific order)

> [!NOTE]
> Some docs collecting common knowledge, usage examples or experiments are available:
> - notebooks
>    - [SSA](docs/notebooks/ssa.ipynb)
> - notes
>    - [ARIMA](docs/notes/arima.pdf)
>    - [RNN](docs/notes/rnn.pdf)
>    - [Singular Spectrum Analysis](docs/notes/ssa.pdf)

## To Do 
- add/integrate models
   - baseline
      - [ ] linear regression and other traditional ML algorithms for tabular data
     -  [ ] ARIMA
      - [ ] vanilla LSTM/GRU
   - others
      - [ ] seasonal ARIMA 
      - [ ] GRU-d (https://arxiv.org/abs/1606.01865)
      - [ ] Dual-attention encoder-decoder RNN (https://www.ijcai.org/proceedings/2017/0366.pdf) 
      - [ ] Bayesian regression with change point (https://osf.io/preprints/osf/fzqxv_v1)
- add/integrate simple transforms
   - [ ] feature engineering
   - [ ] Hilbert-Schmidt Independence Criterion (https://arxiv.org/abs/2305.08529)
- improve clustering and add support for anomaly detection
- add support for end-to-end training (pipelines)
- add simple visualisations
