# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convenience functions for reading data."""

import concurrent.futures
import io
import os
import threading
import zipfile
from alphafold.model import utils
import haiku as hk
import numpy as np
# Internal import (7716).


def _prewarm_jax():
  """Touch the accelerator backend so context init overlaps archive I/O."""
  try:
    import jax
    import jax.numpy as jnp
    jax.block_until_ready(jnp.zeros(8, dtype=jnp.float32))
  except Exception:  # Prewarm is an optimization; never fail the load.
    pass


def get_model_haiku_params(model_name: str, data_dir: str) -> hk.Params:
  """Get the Haiku parameters from a model name."""

  path = os.path.join(data_dir, 'params', f'params_{model_name}.npz')

  warm = threading.Thread(target=_prewarm_jax, daemon=True)
  warm.start()

  # The npz members are stored uncompressed; reading them is a memcpy that
  # releases the GIL, so member reads parallelize across threads.
  with zipfile.ZipFile(path) as zf:
    names = sorted(zf.namelist())

    def worker(batch):
      with zipfile.ZipFile(path) as local:
        return {name[:-4]: np.load(io.BytesIO(local.read(name)),
                                   allow_pickle=False)
                for name in batch}

    batches = [names[i::8] for i in range(8)]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
      parts = list(pool.map(worker, batches))
  params = {}
  for part in parts:
    params.update(part)

  warm.join()
  haiku_params = utils.flat_params_to_haiku(params)
  # Keep only device-resident parameter copies: the host arrays (and the load
  # transients) are freed when this function returns, so they never contribute
  # to the predict-time RSS peak. device_put is a bit-exact fp32 host->device
  # copy, so model numerics are unchanged.
  import jax  # Local import: this module is imported before jax backend init.
  device_params = jax.device_put(haiku_params)
  # Release the host copies and the load transients before returning: the
  # freed blocks would otherwise stay mapped in the allocator and inflate the
  # lifetime peak RSS measured at later stages.
  del params, haiku_params
  from alphafold.model.tf import graph_runtime  # Local: ctypes-only module.
  graph_runtime.release_host_memory()
  return device_params
