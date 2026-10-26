"""Runtime scaling benchmark (Pyro vs fixed torch module) + correctness at scale.

Usage: python bench_scaling.py <pilot|runtime|correctness|equivalence>
Protocol mirrors bench/bench_final.py: n_vars=15000, n_factors=30, n_batch=2,
n_groups=50, 20 epochs, minibatch 256, Adam lr=0.002, seed-42 workload.
"""
import gc
import sys
import time

import numpy as np
import pandas as pd
import pyro
import torch

sys.path.insert(0, "/workspace")
from validate_c2l import (
    SITE_MAP,
    analytic_prior_means,
    compare,
    make_workload_with_truth,
    set_seeds,
    softplus_inv,
)

N_VARS = 15000
N_FACTORS = 30
N_BATCH = 2
N_GROUPS = 50
EPOCHS = 20
BATCH = 256
LR = 2e-3
OUT = "/workspace/bench_out"

import os

os.makedirs(OUT, exist_ok=True)


def get_detection_mean(adata, csd):
    from cell2location.models import Cell2location

    ad = adata.copy()
    Cell2location.setup_anndata(adata=ad, batch_key="batch")
    model = Cell2location(adata=ad, cell_state_df=csd)
    dm = float(model.detection_mean_)
    del model
    pyro.clear_param_store()
    gc.collect()
    torch.cuda.empty_cache()
    return dm


def bench_pyro(adata, csd, epochs=EPOCHS, batch=BATCH, reps=2, seed0=100):
    from cell2location.models import Cell2location

    times, mems = [], []
    for r in range(reps):
        set_seeds(seed0 + r)
        ad = adata.copy()
        Cell2location.setup_anndata(adata=ad, batch_key="batch")
        model = Cell2location(adata=ad, cell_state_df=csd)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train(max_epochs=epochs, batch_size=batch, accelerator="gpu", enable_progress_bar=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        mems.append(torch.cuda.max_memory_allocated() / (1024**2))
        del model, ad
        pyro.clear_param_store()
        gc.collect()
        torch.cuda.empty_cache()
    return times, mems


def bench_torch(adata, csd, detection_mean, epochs=EPOCHS, batch=BATCH, reps=3, seed0=100):
    from cell2location.models._cell2location_torch_module import Cell2locationTorchModule

    x_gpu = torch.tensor(adata.X.toarray(), dtype=torch.bfloat16, device="cuda")
    batch_idx_gpu = torch.tensor(adata.obs["batch"].cat.codes.to_numpy(), dtype=torch.long, device="cuda")
    n_obs = adata.n_obs
    times, mems = [], []
    for r in range(reps):
        set_seeds(seed0 + r)
        module = Cell2locationTorchModule(
            n_obs=n_obs,
            n_vars=N_VARS,
            n_factors=N_FACTORS,
            n_batch=N_BATCH,
            cell_state_mat=csd.values.astype(np.float32),
            n_groups=N_GROUPS,
            detection_mean=detection_mean,
        ).cuda()
        module.init_to_mean()
        module.forward = torch.compile(module.forward, mode="reduce-overhead")
        optimizer = torch.optim.Adam(module.parameters(), lr=LR, fused=True)

        def run_epochs(n):
            for _ in range(n):
                indices = torch.randperm(n_obs, device="cuda")
                for i in range(0, n_obs, batch):
                    bi = indices[i : i + batch]
                    x_batch = x_gpu[bi].float()
                    batch_index = batch_idx_gpu[bi].unsqueeze(-1)
                    optimizer.zero_grad()
                    loss = module(x_batch, bi, batch_index, kl_weight=1.0)
                    loss.backward()
                    optimizer.step()

        run_epochs(2)  # warmup: absorb torch.compile JIT
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_epochs(epochs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        mems.append(torch.cuda.max_memory_allocated() / (1024**2))
        del module, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return times, mems


def run_runtime(sizes, pyro_reps=2, torch_reps=3, tag="runtime"):
    rows = []
    for n_obs in sizes:
        adata, csd, _ = make_workload_with_truth(n_obs=n_obs, n_vars=N_VARS, n_factors=N_FACTORS, n_batch=N_BATCH, seed=42)
        dm = get_detection_mean(adata, csd)
        pt, pm = bench_pyro(adata, csd, reps=pyro_reps)
        print(f"[{tag}] n_obs={n_obs} pyro: {pt} s, {pm} MB", flush=True)
        tt, tm = bench_torch(adata, csd, dm, reps=torch_reps)
        print(f"[{tag}] n_obs={n_obs} torch: {tt} s, {tm} MB", flush=True)
        for r in range(pyro_reps):
            rows.append({"n_obs": n_obs, "impl": "pyro", "rep": r, "time_s": pt[r], "peak_mem_mb": pm[r]})
        for r in range(torch_reps):
            rows.append({"n_obs": n_obs, "impl": "torch_fixed", "rep": r, "time_s": tt[r], "peak_mem_mb": tm[r]})
        pd.DataFrame(rows).to_csv(f"{OUT}/{tag}.csv", index=False)
        del adata, csd
        gc.collect()
    return pd.DataFrame(rows)


def check_init_and_elbo(adata, csd, detection_mean, tiny=1e-6):
    """Init locs/scales vs analytic prior means + ELBO-at-init vs Pyro, on GPU."""
    from pyro.infer import Trace_ELBO
    from scvi import REGISTRY_KEYS
    from torch.distributions import biject_to

    from cell2location.models import Cell2location
    from cell2location.models._cell2location_torch_module import Cell2locationTorchModule

    ad = adata.copy()
    Cell2location.setup_anndata(adata=ad, batch_key="batch")
    set_seeds(0)
    model = Cell2location(adata=ad, cell_state_df=csd)
    model.module.cuda()  # move module buffers to GPU (train() normally handles this)
    am = model.adata_manager
    x = am.get_from_registry(REGISTRY_KEYS.X_KEY)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    b = np.asarray(am.get_from_registry(REGISTRY_KEYS.BATCH_KEY))
    x_t = torch.tensor(np.asarray(x), dtype=torch.float32, device="cuda")
    idx = torch.arange(ad.n_obs, device="cuda")
    b_t = torch.tensor(b, dtype=torch.long, device="cuda")

    guide = model.module.guide
    _ = guide(x_t, idx, b_t)
    transforms = {name: biject_to(site["fn"].support) for name, site in guide.prototype_trace.iter_stochastic_nodes()}

    set_seeds(0)
    tm = Cell2locationTorchModule(
        n_obs=ad.n_obs, n_vars=N_VARS, n_factors=N_FACTORS, n_batch=N_BATCH,
        cell_state_mat=csd.values.astype(np.float32), n_groups=N_GROUPS, detection_mean=detection_mean,
    ).cuda()
    tm.init_to_mean()

    analytic = analytic_prior_means(N_FACTORS, N_GROUPS, detection_mean)
    init_ok, max_loc_err, max_scale_err = True, 0.0, 0.0
    for suffix, site in SITE_MAP:
        pl = transforms[site](getattr(guide.locs, site).detach()).cpu().numpy()
        ps = getattr(guide.scales, site).detach().cpu().numpy()
        tl = torch.exp(getattr(tm, f"locs_{suffix}").detach()).cpu().numpy()
        ts = torch.nn.functional.softplus(getattr(tm, f"scales_{suffix}").detach()).cpu().numpy()
        loc_err = max(float(np.abs(pl - analytic[site]).max()), float(np.abs(tl - analytic[site]).max()))
        scale_err = max(float(np.abs(ps - 0.1).max()), float(np.abs(ts - 0.1).max()))
        max_loc_err, max_scale_err = max(max_loc_err, loc_err), max(max_scale_err, scale_err)
        if loc_err > 1e-3 or scale_err > 1e-4:
            init_ok = False

    # ELBO at init with tiny scales (both sides), full batch
    raw = softplus_inv(tiny)
    for suffix, site in SITE_MAP:
        guide.scales._parameters[f"{site}_unconstrained"].data.fill_(raw)
        with torch.no_grad():
            getattr(tm, f"scales_{suffix}").fill_(raw)
    set_seeds(123)
    elbo = Trace_ELBO(num_particles=1)
    pyro_loss = float(elbo.loss(model.module.model, guide, x_t, idx, b_t))
    with torch.no_grad():
        torch_loss = float(tm(x_t, idx, b_t, kl_weight=1.0).item())
    rel = abs(pyro_loss - torch_loss) / abs(pyro_loss)
    del model, tm
    pyro.clear_param_store()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "init_ok": init_ok, "max_loc_err": max_loc_err, "max_scale_err": max_scale_err,
        "pyro_loss": pyro_loss, "torch_loss": torch_loss, "elbo_rel_diff": rel,
    }


def run_correctness(sizes):
    rows = []
    for n_obs in sizes:
        adata, csd, _ = make_workload_with_truth(n_obs=n_obs, n_vars=N_VARS, n_factors=N_FACTORS, n_batch=N_BATCH, seed=42)
        dm = get_detection_mean(adata, csd)
        res = check_init_and_elbo(adata, csd, dm)
        res["n_obs"] = n_obs
        print(f"[correctness] n_obs={n_obs}: {res}", flush=True)
        rows.append(res)
        pd.DataFrame(rows).to_csv(f"{OUT}/correctness_at_scale.csv", index=False)
        del adata, csd
        gc.collect()
    return pd.DataFrame(rows)


def train_pyro_minibatch(adata, csd, epochs, seed, num_samples=100):
    from cell2location.models import Cell2location

    set_seeds(seed)
    ad = adata.copy()
    Cell2location.setup_anndata(adata=ad, batch_key="batch")
    model = Cell2location(adata=ad, cell_state_df=csd)
    set_seeds(seed)
    model.train(max_epochs=epochs, batch_size=BATCH, accelerator="gpu", enable_progress_bar=False)
    set_seeds(7)
    samples = model.sample_posterior(
        num_samples=num_samples, batch_size=None, return_sites=["w_sf", "m_g", "detection_y_s", "s_g_gene_add", "alpha_g_inverse"]
    )
    pm = {k: v for k, v in samples["post_sample_means"].items()}
    del model, ad
    pyro.clear_param_store()
    gc.collect()
    torch.cuda.empty_cache()
    return pm


def train_torch_minibatch(adata, csd, detection_mean, epochs, seed):
    from cell2location.models._cell2location_torch_module import Cell2locationTorchModule

    set_seeds(seed)
    tm = Cell2locationTorchModule(
        n_obs=adata.n_obs, n_vars=N_VARS, n_factors=N_FACTORS, n_batch=N_BATCH,
        cell_state_mat=csd.values.astype(np.float32), n_groups=N_GROUPS, detection_mean=detection_mean,
    ).cuda()
    tm.init_to_mean()
    x_gpu = torch.tensor(adata.X.toarray(), dtype=torch.bfloat16, device="cuda")
    batch_idx_gpu = torch.tensor(adata.obs["batch"].cat.codes.to_numpy(), dtype=torch.long, device="cuda")
    opt = torch.optim.Adam(tm.parameters(), lr=LR, fused=True)
    n_obs = adata.n_obs
    for _ in range(epochs):
        indices = torch.randperm(n_obs, device="cuda")
        for i in range(0, n_obs, BATCH):
            bi = indices[i : i + BATCH]
            opt.zero_grad()
            loss = tm(x_gpu[bi].float(), bi, batch_idx_gpu[bi].unsqueeze(-1), kl_weight=1.0)
            loss.backward()
            opt.step()
    # posterior means (100 samples)
    set_seeds(7)
    out = {}
    with torch.no_grad():
        for suffix, site in SITE_MAP:
            loc = getattr(tm, f"locs_{suffix}")
            sp_ = torch.nn.functional.softplus(getattr(tm, f"scales_{suffix}"))
            zu = loc.unsqueeze(0) + sp_.unsqueeze(0) * torch.randn(100, *loc.shape, device=loc.device)
            out[site] = torch.exp(zu).mean(0).cpu().numpy()
    del tm, opt
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_equivalence(n_obs=5000, epochs=200):
    adata, csd, truth = make_workload_with_truth(n_obs=n_obs, n_vars=N_VARS, n_factors=N_FACTORS, n_batch=N_BATCH, seed=42)
    dm = get_detection_mean(adata, csd)
    pm_s1 = train_pyro_minibatch(adata, csd, epochs, seed=1)
    print("[equiv] pyro s1 done", flush=True)
    pm_s2 = train_pyro_minibatch(adata, csd, epochs, seed=2)
    print("[equiv] pyro s2 done", flush=True)
    tm_m = train_torch_minibatch(adata, csd, dm, epochs, seed=1)
    print("[equiv] torch fixed done", flush=True)
    rows = []
    for site in ["w_sf", "m_g", "detection_y_s", "s_g_gene_add", "alpha_g_inverse"]:
        floor = compare(pm_s1[site], pm_s2[site])
        fix = compare(tm_m[site], pm_s1[site])
        rows.append({
            "site": site,
            "floor_pearson": floor["pearson"], "fix_pearson": fix["pearson"],
            "floor_relRMSE": floor["rel_rmse"], "fix_relRMSE": fix["rel_rmse"],
        })
    df = pd.DataFrame(rows)
    df["pyro_s1_truth_pearson"] = np.nan
    truth_row = {
        "site": "w_sf_vs_ground_truth",
        "floor_pearson": compare(pm_s2["w_sf"], truth["w_sf"])["pearson"],
        "fix_pearson": compare(tm_m["w_sf"], truth["w_sf"])["pearson"],
        "floor_relRMSE": np.nan,
        "fix_relRMSE": compare(pm_s1["w_sf"], truth["w_sf"])["pearson"],
    }
    df = pd.concat([df, pd.DataFrame([truth_row])], ignore_index=True)
    df.to_csv(f"{OUT}/equivalence_5k.csv", index=False)
    print(df.round(4).to_string(index=False), flush=True)
    np.savez(f"{OUT}/equiv_5k_posteriors.npz", pyro_s1=pm_s1["w_sf"], pyro_s2=pm_s2["w_sf"], torch_fix=tm_m["w_sf"], truth=truth["w_sf"])
    return df


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pilot":
        run_runtime([5000], pyro_reps=1, torch_reps=1, tag="pilot")
    elif cmd == "runtime":
        run_runtime([5000, 10000, 20000], pyro_reps=2, torch_reps=3, tag="runtime")
    elif cmd == "correctness":
        run_correctness([5000, 10000, 20000])
    elif cmd == "equivalence":
        run_equivalence()
