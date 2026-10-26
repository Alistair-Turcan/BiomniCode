# Setup and usage

## Supported setup route

Use Linux x86-64 and a CUDA-13-compatible NVIDIA driver. The evaluated system
used Python 3.12.1, PyTorch 2.13.0+cu130, Triton 3.7.1, CPU JAX 0.10.2, and one
H100 80 GB GPU. Other GPU configurations are described in the inherited kit
documentation, but were not validated in this packaging pass.

Run the build from the repository root, which contains both `af3_torch/` and
`common/`:

```bash
docker build -f af3_torch/environment/Dockerfile -t xfold-optimized:latest .
```

This is the existing Dockerfile's default **source-build** route. It needs
network access to its pinned dependency/source locations and chemical-component
data. It does not require a private prebuilt wheel in the repository.
Do **not** select `WHEELS_FROM=prebuilt` without independently supplying the
exact external artifacts expected by `af3_torch/stock/PINS.json`.

For Apptainer or native virtual environments, follow
[the existing setup guide](../af3_torch/README.md#setup) and
[the complete pinned stack recipe](../af3_torch/STOCK.md#stack).
Their full original text is preserved; inherited setup examples using `fast`
are not the accepted evaluation profile.

## Standard JSON-to-structure entry point

Set `WEIGHTS_DIR` to your own absolute directory; weights are not bundled.
From the repository root:

```bash
WEIGHTS_DIR=/absolute/path/to/your/weights
mkdir -p "$WEIGHTS_DIR" out
docker run --rm -it --gpus all \
  -v "$WEIGHTS_DIR:/weights/af3_torch" \
  -v "$PWD/out:/kit/af3_torch/out" \
  -w /kit/af3_torch xfold-optimized:latest bash
```

Inside that shell, the component's existing commands are:

```bash
bash run.sh install --weights /weights/af3_torch --fetch
export AF3_TORCH_PARAMS_DIR=/weights/af3_torch
bash run.sh check --config h100 --mode off
bash run.sh pred --config h100 --mode off \
  --json_path inputs/1BRS.json --output_dir out/example \
  --num_recycles 10 --diffusion_steps 200
```

`--fetch` downloads/converts the pinned OpenFold3-preview2 checkpoint if needed;
otherwise it checks the existing converted weights. Use the component's
documented input format and substitute your JSON path for the bundled example.
For an external JSON, bind its containing directory into the container and use
that container path. Predictions are written to the mounted `out/` directory.

**Why `off`?** It disables the inherited kit's optional lever sets. The two
accepted optimizations are already source-level changes and remain active when
their inference/CUDA safety guards are satisfied. It does not select the
pre-optimization comparator. Do not use `bash run.sh stock` as a substitute:
that is a separate upstream-CLI route.

The CLI is retained as the existing user-facing JSON/writer interface. The
timed evaluation instead invoked the direct model API below; no CLI, Docker
build, or clean installation was rerun when creating this ZIP.

## Direct API profile used in evaluation

The scientific calls below mirror the completed evaluation's profile without
its timing and telemetry wrapper. They require an **already featurized
`batch.npz`**, the converted weight file, and the pinned torch environment.
The batch is not a raw sequence or JSON input, and is not included here.
Use the pinned CPU featurizer described in the component docs to prepare it.

From the repository root, with `BATCH_NPZ` and `WEIGHTS_FILE` set:

```bash
export BATCH_NPZ=/absolute/path/to/batch.npz
export WEIGHTS_FILE=/absolute/path/to/of3_ported_weights.bin.zst
export PYTHONHASHSEED=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
# Run the following Python with the pinned torch environment, not the JAX one.
```

```python
import os
from pathlib import Path
import sys

assert not any(k.startswith("AF3_TORCH_OPT") for k in os.environ)
sys.path.insert(
    0, str(Path.cwd() / "af3_torch/opt/forward/af3t/af3_torch")
)
import torch
import af3_torch_api as A
from xfold.fastnn import config

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
config.layer_norm_implementation = "triton"
config.dot_product_attention_implementation = "triton"
config.gated_linear_unit_implementation = "triton"

model = A.build_model(
    os.environ["WEIGHTS_FILE"], levers=(),
    num_recycles=10, num_samples=5, diffusion_steps=200,
)
batch = A.batch_from_npz(os.environ["BATCH_NPZ"])
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
model.diffusion_head.clear_static()
with A.inference():
    embeddings = A.run_trunk(model, batch)
    coordinates = A.run_diffusion(model, batch, embeddings, seed=seed)
    confidence = [
        A.run_confidence(model, batch, embeddings, coordinates[i])
        for i in range(5)
    ]
    distogram = A.run_distogram(model, batch, embeddings)
```

This is an API usage example, not a complete scientific-output writer or timing
harness. It intentionally avoids enabling `exact`, `fast`, `big`, sample
batching, weight conversion to bf16, or reduced steps/samples.
The original evaluation additionally fixed CPU affinity and monitored runtime,
memory, RNG states, and native outputs in an external harness.

## Reproducibility and validation

- [Source identities and external weight checksums](../provenance/source_pins.json)
  distinguish the accepted tree from its upstream repositories.
- [The accepted patch](../provenance/accepted.patch) is already applied.
  Its baseline is the corrected preview2 port, not pristine upstream xfold.
- The original source inventory and root `SHA256SUMS` support byte-level checks.
- The source-built featurizer, rebuilt chemical-component data, compiler
  products, GPU hardware, and driver can differ from the historical evaluation
  environment. Exact replay requires restoring the recorded external artifacts
  and environment; this ZIP does not contain them.
- [Packaging validation](../provenance/package_validation.json) records static
  checks only. No new installation, numerical correctness run, GPU inference,
  or performance measurement is claimed by this package.

Review the component's existing tests before making further changes. Some
inherited tests refer to the original kit's source/accounting assumptions or
optional providers; they are retained unchanged, not represented as an
entire test suite newly passed for this accepted patch.
