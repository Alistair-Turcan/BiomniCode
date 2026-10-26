"""Benchmark the Pyro-free Cell2location module with all optimizations.
Usage: python bench_final.py <n_obs> <n_epochs> <reps>
"""
import sys, time, gc
import numpy as np, torch, pyro
sys.path.insert(0, "..")
from cell2location.models._cell2location_torch_module import Cell2locationTorchModule
from workload_gen import make_workload
import random
from scvi import REGISTRY_KEYS

def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); pyro.set_rng_seed(seed)

n_obs = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3

set_all_seeds(42)
adata, csd = make_workload(n_obs=n_obs, n_vars=15000, n_factors=30, n_batch=2, seed=42)
from cell2location.models import Cell2location
Cell2location.setup_anndata(adata=adata, batch_key="batch")
model = Cell2location(adata=adata, cell_state_df=csd)
detection_mean = float(model.detection_mean_)
am = model.adata_manager
x_data = am.get_from_registry(REGISTRY_KEYS.X_KEY)
batch_idx = am.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
pyro.clear_param_store(); del model; gc.collect(); torch.cuda.empty_cache()

# bfloat16 data preloading (halves memory, same range as float32)
x_gpu = torch.tensor(x_data.toarray(), dtype=torch.bfloat16, device='cuda')
batch_idx_gpu = torch.tensor(batch_idx, dtype=torch.long, device='cuda')

times = []; gpu_mems = []
for rep in range(reps):
    set_all_seeds(42 + rep)
    module = Cell2locationTorchModule(
        n_obs=n_obs, n_vars=15000, n_factors=30, n_batch=2,
        cell_state_mat=csd.values.astype(np.float32),
        n_groups=50, detection_mean=detection_mean,
    ).cuda()
    module.init_to_mean()
    # torch.compile fuses all elementwise ops into Triton kernels
    module.forward = torch.compile(module.forward, mode="reduce-overhead")
    # Fused Adam: single CUDA kernel for optimizer step
    optimizer = torch.optim.Adam(module.parameters(), lr=0.002, fused=True)
    
    # Warmup (2 epochs for torch.compile JIT compilation)
    for _ in range(2):
        indices = torch.randperm(n_obs, device='cuda')
        for i in range(0, n_obs, 256):
            bi = indices[i:i+256]
            x_batch = x_gpu[bi].float(); batch_index = batch_idx_gpu[bi].unsqueeze(-1)
            optimizer.zero_grad(); loss = module(x_batch, bi, batch_index, kl_weight=1.0)
            loss.backward(); optimizer.step()
    
    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        indices = torch.randperm(n_obs, device='cuda')
        for i in range(0, n_obs, 256):
            bi = indices[i:i+256]
            x_batch = x_gpu[bi].float(); batch_index = batch_idx_gpu[bi].unsqueeze(-1)
            optimizer.zero_grad(); loss = module(x_batch, bi, batch_index, kl_weight=1.0)
            loss.backward(); optimizer.step()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
    gpu_mems.append(torch.cuda.max_memory_allocated()/(1024**2))
    del module, optimizer; gc.collect(); torch.cuda.empty_cache()

print(f"RESULT {n_obs} {n_epochs} {np.mean(times):.3f} {np.mean(gpu_mems):.1f} {np.std(times):.3f}")
gc.collect(); torch.cuda.empty_cache()
