# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Child-process entry point for isolated cached feature-graph execution.

Runs a cached frozen featurization graph in this (short-lived) process so the
graph's allocator high-water mark is charged to this process and released by
exit, instead of raising the caller's lifetime peak RSS. Numerics are identical
to an in-process run: same graph bytes, same native library, same session
configuration.

Usage: python -m alphafold.model.tf.graph_runtime_worker CACHE_PATH OUT_PKL
Exit codes: 0 = outputs pickled to OUT_PKL; 3 = envelope invalid;
4 = native run declined/failed.
"""

import hashlib
import json
import pickle
import struct
import sys

from alphafold.model.tf import graph_runtime


def _read_envelope(path):
  """Parse a cached feature-graph envelope (mirrors features._read_feature_graph)."""
  try:
    with open(path, 'rb') as f:
      envelope = f.read()
    payload = envelope[32:]
    if envelope[:32] != hashlib.sha256(payload).digest():
      return None
    metadata_size = struct.unpack('>Q', payload[:8])[0]
    metadata = json.loads(payload[8:8 + metadata_size])
    graph_bytes = payload[8 + metadata_size:]
    if metadata['identity'] != path.rsplit('/', 1)[-1].rsplit('.', 1)[0]:
      return None
    if not isinstance(metadata['seed_calls'], int) or metadata['seed_calls'] < 0:
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


def main():
  cache_path, out_path = sys.argv[1], sys.argv[2]
  parsed = _read_envelope(cache_path)
  if parsed is None:
    sys.exit(3)
  metadata, graph_bytes = parsed
  result = graph_runtime.run(graph_bytes, metadata['outputs'])
  if result is None:
    sys.exit(4)
  with open(out_path, 'wb') as f:
    pickle.dump(result, f, protocol=4)
  sys.exit(0)


if __name__ == '__main__':
  main()
