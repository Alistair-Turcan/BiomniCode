import numpy as np, pandas as pd, scipy.sparse as sp
from anndata import AnnData
def make_workload(n_obs=10000, n_vars=15000, n_factors=30, n_batch=2, seed=42):
    rng = np.random.default_rng(seed)
    cell_state = rng.gamma(shape=0.3, scale=3.0, size=(n_vars, n_factors)).astype(np.float32)
    for f in range(n_factors):
        markers = rng.choice(n_vars, size=50, replace=False)
        cell_state[markers, f] = rng.gamma(shape=5.0, scale=10.0, size=50).astype(np.float32)
    gene_names = [f"Gene_{i}" for i in range(n_vars)]
    cell_type_names = [f"CellType_{f}" for f in range(n_factors)]
    cell_state_df = pd.DataFrame(cell_state, index=gene_names, columns=cell_type_names)
    w_sf = rng.gamma(shape=2.0, scale=4.0, size=(n_obs, n_factors)).astype(np.float32)
    mu = w_sf @ cell_state.T
    batch = rng.integers(0, n_batch, size=n_obs)
    detection = rng.gamma(shape=20.0, scale=0.5, size=(n_batch,)).astype(np.float32)
    y_s = detection[batch] * rng.gamma(shape=20.0, scale=1.0/20.0, size=n_obs).astype(np.float32)
    background = rng.gamma(shape=1.0, scale=0.5, size=(n_batch, n_vars)).astype(np.float32)
    mu_full = (mu + background[batch]) * y_s[:, None]
    alpha = rng.gamma(shape=9.0, scale=1.0/3.0, size=(n_batch, n_vars)).astype(np.float32)
    alpha_obs = alpha[batch]
    rate = rng.gamma(alpha_obs, 1.0 / (alpha_obs / np.maximum(mu_full, 1e-6)))
    counts = np.maximum(rng.poisson(rate).astype(np.float32), 0).astype(np.float32)
    X = sp.csr_matrix(counts)
    obs = pd.DataFrame({"batch": pd.Categorical(batch.astype(str))}, index=[f"Spot_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=gene_names)
    return AnnData(X=X, obs=obs, var=var), cell_state_df
