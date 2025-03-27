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

### To Do 
- add/integrate models
   - baseline
      - [ ] linear regression and other traditional ML algorithms for tabular data
     -  [ ] Arima
      - [ ] vanilla Lstm/Gru
   - others
      - [ ] seasonal Arima 
      - [ ] Gru-d (https://arxiv.org/abs/1606.01865)
      - [ ] Dual-attention encoder-decoder Rnn (https://www.ijcai.org/proceedings/2017/0366.pdf) 
      - [ ] Bayesian regression with change point (https://osf.io/preprints/osf/fzqxv_v1)
- add/integrate simple transforms
   - [ ] feature engineering
   - [ ] Hilbert-Schmidt Independence Criterion (https://arxiv.org/abs/2305.08529)
- improve clustering and add support for anomaly detection
- add support for end-to-end training (pipelines)
- add simple visualisations

### Docs
- notebooks
   - [SSA](docs/notebooks/ssa.ipynb)
- notes
   - [ARIMA](docs/notes/arima.pdf)
   - [RNN](docs/notes/rnn.pdf)
   - [Singular Spectrum Analysis](docs/notes/ssa.pdf)
