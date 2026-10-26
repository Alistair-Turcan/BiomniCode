"""Validation of the cell2location Pyro-free torch module initialization fix.

Parts:
  A: deterministic initialization check — torch module (fixed & unfixed) vs Pyro
     AutoNormal guide (init_to_mean) vs analytic prior means, all 17 latent sites.
  B: ELBO equivalence at init — torch loss vs Pyro Trace_ELBO loss with tiny scales.
  C: end-to-end training equivalence — posterior means after training, torch
     (fixed & unfixed) vs Pyro original (2 seeds = noise floor) vs ground truth.
"""
import importlib.util
import math
import sys
import time

import numpy as np
import pandas as pd
import pyro
import scipy.sparse as sp
import torch
from anndata import AnnData

UNFIXED_MODULE_PATH = "/workspace/c2l_inspect/cell2location_packaged/cell2location/models/_cell2location_torch_module.py"

# (torch attr suffix, pyro site name) — all 17 latent sites
SITE_MAP = [
    ("m_g_mean", "m_g_mean"),
    ("m_g_alpha_e_inv", "m_g_alpha_e_inv"),
    ("m_g", "m_g"),
    ("k_r", "k_r_factors_per_groups"),
    ("x_fr", "x_fr_group2fact"),
    ("dmye", "detection_mean_y_e"),
    ("sgah", "s_g_gene_add_alpha_hyp"),
    ("sgm", "s_g_gene_add_mean"),
    ("sgai", "s_g_gene_add_alpha_e_inv"),
    ("sga", "s_g_gene_add"),
    ("agph", "alpha_g_phi_hyp"),
    ("agi", "alpha_g_inverse"),
    ("nsc", "n_s_cells_per_location"),
    ("bsg", "b_s_groups_per_location"),
    ("zsr", "z_sr_groups_factors"),
    ("wsf", "w_sf"),
    ("dys", "detection_y_s"),
]

N_CELLS = 8.0
DETECTION_ALPHA = 200.0
A_FACTORS = 7.0
B_GROUPS = 7.0


def set_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


def softplus_inv(y):
    return math.log(math.expm1(y))


def make_workload_with_truth(n_obs=1000, n_vars=1000, n_factors=8, n_batch=2, seed=42):
    """bench/workload_gen.py replicated verbatim, additionally returning ground truth."""
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
    y_s = detection[batch] * rng.gamma(shape=20.0, scale=1.0 / 20.0, size=n_obs).astype(np.float32)
    background = rng.gamma(shape=1.0, scale=0.5, size=(n_batch, n_vars)).astype(np.float32)
    mu_full = (mu + background[batch]) * y_s[:, None]
    alpha = rng.gamma(shape=9.0, scale=1.0 / 3.0, size=(n_batch, n_vars)).astype(np.float32)
    alpha_obs = alpha[batch]
    rate = rng.gamma(alpha_obs, 1.0 / (alpha_obs / np.maximum(mu_full, 1e-6)))
    counts = np.maximum(rng.poisson(rate).astype(np.float32), 0).astype(np.float32)
    X = sp.csr_matrix(counts)
    obs = pd.DataFrame({"batch": pd.Categorical(batch.astype(str))}, index=[f"Spot_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=gene_names)
    truth = {"w_sf": w_sf, "y_s": y_s, "batch": batch}
    return AnnData(X=X, obs=obs, var=var), cell_state_df, truth


def load_torch_module_cls(fixed=True):
    if fixed:
        from cell2location.models._cell2location_torch_module import Cell2locationTorchModule

        return Cell2locationTorchModule
    spec = importlib.util.spec_from_file_location("c2l_torch_unfixed", UNFIXED_MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Cell2locationTorchModule


def build_pyro_model(adata, csd, n_groups, seed=0):
    from cell2location.models import Cell2location

    set_seeds(seed)
    Cell2location.setup_anndata(adata=adata, batch_key="batch")
    model = Cell2location(
        adata,
        cell_state_df=csd,
        n_groups=n_groups,
        N_cells_per_location=N_CELLS,
        detection_alpha=DETECTION_ALPHA,
    )
    return model


def get_batch_tensors(model):
    from scvi import REGISTRY_KEYS

    am = model.adata_manager
    x = am.get_from_registry(REGISTRY_KEYS.X_KEY)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    b = np.asarray(am.get_from_registry(REGISTRY_KEYS.BATCH_KEY))
    n_obs = x.shape[0]
    return (
        torch.tensor(np.asarray(x), dtype=torch.float32),
        torch.arange(n_obs),
        torch.tensor(b, dtype=torch.long),
    )


def build_torch_module(adata, csd, n_groups, detection_mean, fixed=True, seed=0):
    cls = load_torch_module_cls(fixed=fixed)
    set_seeds(seed)
    tm = cls(
        n_obs=adata.n_obs,
        n_vars=adata.n_vars,
        n_factors=csd.shape[1],
        n_batch=adata.obs["batch"].nunique(),
        cell_state_mat=csd.values.astype(np.float32),
        n_groups=n_groups,
        detection_mean=detection_mean,
        detection_alpha=DETECTION_ALPHA,
        N_cells_per_location=N_CELLS,
    )
    tm.init_to_mean()
    return tm


def analytic_prior_means(n_factors, n_groups, detection_mean):
    return {
        "m_g_mean": 1.0,
        "m_g_alpha_e_inv": 1.0 / 3.0,
        "m_g": 1.0,
        "k_r_factors_per_groups": A_FACTORS / B_GROUPS,
        "x_fr_group2fact": 1.0 / n_factors,
        "detection_mean_y_e": detection_mean,
        "detection_y_s": detection_mean,
        "s_g_gene_add_alpha_hyp": 3.0,
        "s_g_gene_add_mean": 0.01,
        "s_g_gene_add_alpha_e_inv": 1.0 / 3.0,
        "s_g_gene_add": 0.01,
        "alpha_g_phi_hyp": 3.0,
        "alpha_g_inverse": 1.0 / 3.0,
        "n_s_cells_per_location": N_CELLS,
        "b_s_groups_per_location": B_GROUPS,
        "z_sr_groups_factors": N_CELLS / n_groups,
        "w_sf": N_CELLS / n_factors,
    }


def pyro_guide_init_params(model, batch):
    """Run guide once to trigger prototype setup, return per-site loc (unconstrained),
    loc transformed to constrained space via the site's actual biject_to transform,
    and constrained scale."""
    from torch.distributions import biject_to

    x, idx, b = batch
    guide = model.module.guide
    _ = guide(x, idx, b)
    transforms = {}
    for name, site in guide.prototype_trace.iter_stochastic_nodes():
        transforms[name] = biject_to(site["fn"].support)
    out = {}
    for suffix, site in SITE_MAP:
        loc_u = getattr(guide.locs, site).detach().clone()
        scale_c = getattr(guide.scales, site).detach().clone()
        loc_c = transforms[site](loc_u)
        out[site] = {"loc_unconstr": loc_u.numpy(), "loc_constr": loc_c.numpy(), "scale_constr": scale_c.numpy()}
    return out


def torch_init_params(tm):
    out = {}
    for suffix, site in SITE_MAP:
        loc = getattr(tm, f"locs_{suffix}").detach().clone()
        sc = getattr(tm, f"scales_{suffix}").detach().clone()
        out[site] = {
            "loc_unconstr": loc.numpy(),
            "loc_constr": torch.exp(loc).numpy(),
            "scale_constr": torch.nn.functional.softplus(sc).numpy(),
        }
    return out


def part_a(n_obs=200, n_vars=300, n_factors=8, n_groups=20, seed=0):
    adata, csd, truth = make_workload_with_truth(n_obs=n_obs, n_vars=n_vars, n_factors=n_factors, seed=42)
    model = build_pyro_model(adata.copy(), csd, n_groups, seed=seed)
    batch = get_batch_tensors(model)
    detection_mean = float(model.detection_mean_)
    pyro_params = pyro_guide_init_params(model, batch)
    tm_fixed = build_torch_module(adata, csd, n_groups, detection_mean, fixed=True, seed=seed)
    tm_unfixed = build_torch_module(adata, csd, n_groups, detection_mean, fixed=False, seed=seed)
    fixed_params = torch_init_params(tm_fixed)
    unfixed_params = torch_init_params(tm_unfixed)
    analytic = analytic_prior_means(n_factors, n_groups, detection_mean)

    rows = []
    for suffix, site in SITE_MAP:
        pv = np.unique(pyro_params[site]["loc_constr"].round(6))
        fv = np.unique(fixed_params[site]["loc_constr"].round(6))
        uv = np.unique(unfixed_params[site]["loc_constr"].round(6))
        av = analytic[site]
        ps = np.unique(pyro_params[site]["scale_constr"].round(8))
        fs = np.unique(fixed_params[site]["scale_constr"].round(8))
        us = np.unique(unfixed_params[site]["scale_constr"].round(8))
        rows.append(
            {
                "site": site,
                "analytic_mean": av,
                "pyro_loc_constr": pv.tolist() if len(pv) <= 3 else f"{len(pv)} uniq",
                "torch_fixed_loc_constr": fv.tolist() if len(fv) <= 3 else f"{len(fv)} uniq",
                "torch_unfixed_loc_constr": uv.tolist() if len(uv) <= 3 else f"{len(uv)} uniq",
                "loc_match": bool(np.allclose(pv, fv, rtol=1e-4, atol=1e-6) and np.allclose(pv, av, rtol=1e-3, atol=1e-6)),
                "pyro_scale": ps.tolist() if len(ps) <= 3 else f"{len(ps)} uniq",
                "torch_fixed_scale": fs.tolist() if len(fs) <= 3 else f"{len(fs)} uniq",
                "torch_unfixed_scale": us.tolist() if len(us) <= 3 else f"{len(us)} uniq",
                "scale_match": bool(np.allclose(ps, fs, rtol=1e-4, atol=1e-8)),
            }
        )
    df = pd.DataFrame(rows)
    return df, {"model": model, "batch": batch, "tm_fixed": tm_fixed, "tm_unfixed": tm_unfixed, "detection_mean": detection_mean}


def set_pyro_guide_scales(model, value):
    """Set all guide scales (constrained) to `value` in place."""
    guide = model.module.guide
    raw_target = softplus_inv(value)
    for suffix, site in SITE_MAP:
        raw = guide.scales._parameters[f"{site}_unconstrained"]
        raw.data.fill_(raw_target)
        # sanity: constrained readback
        got = getattr(guide.scales, site).detach()
        assert torch.allclose(got, torch.full_like(got, value), rtol=1e-3), (site, got.flatten()[:3])


def set_torch_scales(tm, value):
    with torch.no_grad():
        for suffix, site in SITE_MAP:
            getattr(tm, f"scales_{suffix}").fill_(softplus_inv(value))


def part_b(model, batch, tm_fixed, tiny=1e-6):
    from pyro.infer import Trace_ELBO

    x, idx, b = batch
    set_seeds(123)
    set_pyro_guide_scales(model, tiny)
    set_torch_scales(tm_fixed, tiny)
    elbo = Trace_ELBO(num_particles=1)
    pyro_loss = elbo.loss(model.module.model, model.module.guide, x, idx, b)
    with torch.no_grad():
        torch_loss = tm_fixed(x, idx, b, kl_weight=1.0).item()
    return {"pyro_loss": float(pyro_loss), "torch_loss": float(torch_loss)}


# ---------------- Part C: end-to-end training equivalence ----------------


def train_pyro(adata, csd, n_groups, epochs, seed):
    model = build_pyro_model(adata.copy(), csd, n_groups, seed=seed)
    set_seeds(seed)
    t0 = time.time()
    model.train(max_epochs=epochs, accelerator="cpu", enable_progress_bar=False)
    return model, time.time() - t0


def train_torch(adata, csd, n_groups, detection_mean, epochs, seed, fixed=True, lr=2e-3):
    tm = build_torch_module(adata, csd, n_groups, detection_mean, fixed=fixed, seed=seed)
    x = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    b = torch.tensor(adata.obs["batch"].cat.codes.to_numpy(), dtype=torch.long).unsqueeze(-1)
    idx = torch.arange(adata.n_obs)
    opt = torch.optim.Adam(tm.parameters(), lr=lr)
    set_seeds(seed + 1000)
    t0 = time.time()
    losses = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = tm(x, idx, b, kl_weight=1.0)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return tm, time.time() - t0, losses


@torch.no_grad()
def torch_posterior_means(tm, n_samples=500, seed=7):
    set_seeds(seed)
    out = {}
    for suffix, site in SITE_MAP:
        loc = getattr(tm, f"locs_{suffix}")
        sp = torch.nn.functional.softplus(getattr(tm, f"scales_{suffix}"))
        zu = loc.unsqueeze(0) + sp.unsqueeze(0) * torch.randn(n_samples, *loc.shape)
        out[site] = torch.exp(zu).mean(0).numpy()
    return out


POSTERIOR_SITES = ["w_sf", "m_g", "detection_y_s", "s_g_gene_add", "alpha_g_inverse"]


def pyro_posterior_means(model, n_samples=500, seed=7):
    set_seeds(seed)
    samples = model.sample_posterior(num_samples=n_samples, batch_size=None, return_sites=POSTERIOR_SITES)
    return {k: v for k, v in samples["post_sample_means"].items()}


def compare(a, b):
    from scipy.stats import pearsonr, spearmanr

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return {
        "pearson": float(pearsonr(a, b)[0]),
        "spearman": float(spearmanr(a, b)[0]),
        "rel_rmse": float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)),
    }


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "a"
    if part == "a":
        df, _ = part_a()
        print(df.to_string())
    elif part == "b":
        df, ctx = part_a()
        res = part_b(ctx["model"], ctx["batch"], ctx["tm_fixed"])
        print(res)
