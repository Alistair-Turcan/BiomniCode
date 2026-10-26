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

"""Code to generate processed features."""
import copy
import hashlib
import importlib
import json
import numbers
import os
from pathlib import Path
import pickle
import struct
import sys
import tempfile
from typing import List, Mapping, Tuple

from alphafold.model.tf import graph_runtime
from alphafold.model.tf import utils as tf_utils
from google.protobuf.message import DecodeError
import ml_collections
import numpy as np

FeatureDict = Mapping[str, np.ndarray]
__all__ = [
    'copy', 'hashlib', 'json', 'numbers', 'os', 'Path', 'pickle', 'struct',
    'tempfile', 'List', 'Mapping', 'Tuple', 'input_pipeline', 'proteins_dataset',
    'tf_utils', 'DecodeError', 'ml_collections', 'np', 'tf', 'FeatureDict',
    'make_data_config', 'np_example_to_features',
]


def __getattr__(name):
  modules = {
      'tf': 'tensorflow.compat.v1',
      'input_pipeline': 'alphafold.model.tf.input_pipeline',
      'proteins_dataset': 'alphafold.model.tf.proteins_dataset',
  }
  if name in modules:
    module = importlib.import_module(modules[name])
    globals()[name] = module
    return module
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
  return sorted(set(globals()) | {'tf', 'input_pipeline', 'proteins_dataset'})


def _load_python_pipeline():
  for name in ('tf', 'input_pipeline', 'proteins_dataset'):
    if name not in globals():
      __getattr__(name)


def _feature_graph_cache_path(np_example, cfg, random_seed):
  """Identify compiled preprocessing code, never processed feature outputs."""
  if not isinstance(random_seed, numbers.Integral):
    return None
  if not isinstance(tf_utils.seed_maker.next_seed, numbers.Integral):
    return None
  for value in np_example.values():
    if not isinstance(value, np.ndarray):
      return None
    if value.dtype.hasobject and not all(
        isinstance(item, (bytes, str)) for item in value.flat
    ):
      return None
  try:
    installed = graph_runtime.distribution()
    tensorflow_version = (tf.__version__ if 'tf' in globals() else
                          installed[0] if installed else None)
    if tensorflow_version is None:
      return None
    identity = hashlib.sha256()
    identity.update(b'alphafold-feature-graph-v1')
    identity.update(json.dumps({
        'tensorflow': tensorflow_version, 'numpy': np.__version__,
        'random_seed': int(random_seed),
        'seed_counter': int(tf_utils.seed_maker.next_seed),
        'config': cfg.to_dict(),
    }, sort_keys=True).encode())
    identity.update(pickle.dumps(np_example, protocol=4))
    source_root = Path(__file__).resolve().parents[1]
    for source in sorted(source_root.rglob('*.py')):
      identity.update(json.dumps([
          str(source.relative_to(source_root)),
          hashlib.sha256(source.read_bytes()).hexdigest(),
      ]).encode())
    cache_home = Path(os.environ.get(
        'XDG_CACHE_HOME', os.path.expanduser('~/.cache')
    ))
    return cache_home / 'alphafold/feature_graphs' / (identity.hexdigest() + '.graph')
  except (OSError, TypeError, ValueError):
    return None


def _read_feature_graph(path):
  if path is None:
    return None
  try:
    envelope = path.read_bytes()
    payload = envelope[32:]
    if envelope[:32] != hashlib.sha256(payload).digest():
      return None
    metadata_size = struct.unpack('>Q', payload[:8])[0]
    metadata = json.loads(payload[8:8 + metadata_size])
    graph_bytes = payload[8 + metadata_size:]
    if metadata['identity'] != path.stem:
      return None
    seed_calls = metadata['seed_calls']
    if not isinstance(seed_calls, int) or seed_calls < 0:
      return None
    if metadata['sha256'] != hashlib.sha256(graph_bytes).hexdigest():
      return None
    if not isinstance(metadata['outputs'], dict):
      return None
    if not all(isinstance(k, str) and isinstance(v, str)
               for k, v in metadata['outputs'].items()):
      return None
    return metadata, graph_bytes
  except (OSError, ValueError, TypeError, KeyError, struct.error):
    return None


def _load_feature_graph(path, random_seed):
  cached = _read_feature_graph(path)
  if cached is None:
    return None
  metadata, graph_bytes = cached
  try:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graph_bytes)
    graph = tf.Graph()
    with graph.as_default(), tf.device('/device:CPU:0'):
      tf.compat.v1.set_random_seed(random_seed)
      tf.import_graph_def(graph_def, name='')
    batch = {key: graph.get_tensor_by_name(name)
             for key, name in metadata['outputs'].items()}
    graph.finalize()
    tf_utils.seed_maker.next_seed += metadata['seed_calls']
    return graph, batch
  except (OSError, ValueError, TypeError, KeyError, struct.error,
          tf.errors.OpError, DecodeError):
    return None


def _save_feature_graph(path, graph, batch, seed_calls):
  if path is None:
    return
  temporary = None
  try:
    graph_bytes = graph.as_graph_def().SerializeToString()
    metadata = json.dumps({
        'outputs': {key: value.name for key, value in batch.items()},
        'identity': path.stem,
        'seed_calls': int(seed_calls),
        'sha256': hashlib.sha256(graph_bytes).hexdigest(),
    }).encode()
    payload = struct.pack('>Q', len(metadata)) + metadata + graph_bytes
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
      temporary = stream.name
      stream.write(hashlib.sha256(payload).digest())
      stream.write(payload)
    os.replace(temporary, path)
    temporary = None
  except OSError:
    pass  # An unavailable cache must not change the preprocessing API.
  finally:
    if temporary is not None:
      try:
        os.unlink(temporary)
      except OSError:
        pass


def make_data_config(
    config: ml_collections.ConfigDict,
    num_res: int,
) -> Tuple[ml_collections.ConfigDict, List[str]]:
  """Makes a data config for the input pipeline."""
  cfg = copy.deepcopy(config.data)

  feature_names = cfg.common.unsupervised_features
  if cfg.common.use_templates:
    feature_names += cfg.common.template_features

  with cfg.unlocked():
    cfg.eval.crop_size = num_res

  return cfg, feature_names


def np_example_to_features(
    np_example: FeatureDict,
    config: ml_collections.ConfigDict,
    random_seed: int = 0,
) -> FeatureDict:
  """Preprocesses NumPy feature dict using TF pipeline."""
  np_example = dict(np_example)
  num_res = int(np_example['seq_length'][0])
  cfg, feature_names = make_data_config(config, num_res=num_res)

  if 'deletion_matrix_int' in np_example:
    np_example['deletion_matrix'] = np_example.pop(
        'deletion_matrix_int'
    ).astype(np.float32)

  seed_counter_before = tf_utils.seed_maker.next_seed
  cache_path = _feature_graph_cache_path(np_example, cfg, random_seed)
  envelope = _read_feature_graph(cache_path)
  # Already-loaded TensorFlow may carry application-level Session settings.
  # Preserve that Python path rather than assuming its global configuration.
  if envelope is not None and 'tensorflow' not in sys.modules:
    metadata, graph_bytes = envelope
    # Run the cached graph in a child process when possible: the graph's
    # allocator high-water mark is then charged to the child and released on
    # exit instead of raising this process's lifetime peak RSS. Outputs are
    # bit-identical (same graph bytes, native library, and session config).
    native_features = graph_runtime.run_isolated(cache_path)
    if native_features is None:
      native_features = graph_runtime.run(graph_bytes, metadata['outputs'])
    if native_features is not None:
      # Release unpickle/transient blocks held by the allocator so the
      # predict-time and serialize-time peaks start from a lower base.
      graph_runtime.release_host_memory()
      tf_utils.seed_maker.next_seed += metadata['seed_calls']
      return native_features
  _load_python_pipeline()
  cached = _load_feature_graph(cache_path, random_seed)
  if cached is None:
    tf_graph = tf.Graph()
    with tf_graph.as_default(), tf.device('/device:CPU:0'):
      tf.compat.v1.set_random_seed(random_seed)
      tensor_dict = proteins_dataset.np_to_tensor_dict(
          np_example=np_example, features=feature_names
      )
      processed_batch = input_pipeline.process_tensors_from_config(
          tensor_dict, cfg
      )
    tf_graph.finalize()
    if cache_path is not None:
      _save_feature_graph(
          cache_path, tf_graph, processed_batch,
          tf_utils.seed_maker.next_seed - seed_counter_before,
      )
  else:
    tf_graph, processed_batch = cached

  with tf.Session(graph=tf_graph) as sess:
    features = sess.run(processed_batch)

  return {k: v for k, v in features.items() if v.dtype != 'O'}
