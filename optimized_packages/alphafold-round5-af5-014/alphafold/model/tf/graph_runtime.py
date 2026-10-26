# Copyright 2026
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Execute serialized feature graphs without loading TensorFlow's Python API."""

import ctypes
import functools
from importlib import metadata

import numpy as np


class _Output(ctypes.Structure):
  _fields_ = [('oper', ctypes.c_void_p), ('index', ctypes.c_int)]


# Numeric TF_DataType values from tensorflow/c/tf_datatype.h. Strings are
# fetched, but omitted like the existing preprocessing object's dtype filter.
_DTYPES = {
    1: np.float32, 2: np.float64, 3: np.int32, 4: np.uint8,
    5: np.int16, 6: np.int8, 8: np.complex64, 9: np.int64,
    10: np.bool_, 17: np.uint16, 18: np.complex128, 19: np.float16,
    22: np.uint32, 23: np.uint64,
}


# Serialized ConfigProto(intra_op_parallelism_threads=1,
# inter_op_parallelism_threads=8). Intra-op stays at 1 so every op keeps the
# harness's pinned single-threaded reduction order; inter-op threads only
# schedule independent ops concurrently, which cannot change any op's output
# bytes (stateful RNG ordering is unaffected: it is serialized by the
# sequential predraw loop's parallel_iterations=1, not by the inter-op pool).
_SESSION_CONFIG = b'\x10\x01\x28\x08'


def distribution():
  """Find the native library from the installed TF distribution, or decline."""
  for name in ('tensorflow-cpu', 'tensorflow'):
    try:
      dist = metadata.distribution(name)
      path = dist.locate_file('tensorflow/libtensorflow_cc.so.2')
      if path.is_file():
        return dist.version, str(path)
    except metadata.PackageNotFoundError:
      continue
  return None


@functools.lru_cache(maxsize=1)
def _library():
  installed = distribution()
  if installed is None:
    return None
  try:
    lib = ctypes.CDLL(installed[1])
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    signatures = {
        'TF_NewStatus': (ptr, []),
        'TF_DeleteStatus': (None, [ptr]),
        'TF_GetCode': (integer, [ptr]),
        'TF_NewGraph': (ptr, []),
        'TF_DeleteGraph': (None, [ptr]),
        'TF_NewBufferFromString': (ptr, [ptr, ctypes.c_size_t]),
        'TF_DeleteBuffer': (None, [ptr]),
        'TF_NewImportGraphDefOptions': (ptr, []),
        'TF_DeleteImportGraphDefOptions': (None, [ptr]),
        'TF_GraphImportGraphDef': (None, [ptr, ptr, ptr, ptr]),
        'TF_GraphOperationByName': (ptr, [ptr, ctypes.c_char_p]),
        'TF_OperationNumOutputs': (integer, [ptr]),
        'TF_OperationOutputType': (integer, [_Output]),
        'TF_NewSessionOptions': (ptr, []),
        'TF_SetConfig': (None, [ptr, ptr, ctypes.c_size_t, ptr]),
        'TF_DeleteSessionOptions': (None, [ptr]),
        'TF_NewSession': (ptr, [ptr, ptr, ptr]),
        'TF_CloseSession': (None, [ptr, ptr]),
        'TF_DeleteSession': (None, [ptr, ptr]),
        'TF_DeleteTensor': (None, [ptr]),
        'TF_TensorType': (integer, [ptr]),
        'TF_NumDims': (integer, [ptr]),
        'TF_Dim': (ctypes.c_int64, [ptr, integer]),
        'TF_TensorByteSize': (ctypes.c_size_t, [ptr]),
        'TF_TensorData': (ptr, [ptr]),
        'TF_SessionRun': (None, [
            ptr, ptr, ctypes.POINTER(_Output), ctypes.POINTER(ptr), integer,
            ctypes.POINTER(_Output), ctypes.POINTER(ptr), integer,
            ctypes.POINTER(ptr), integer, ptr, ptr,
        ]),
    }
    for name, (result, args) in signatures.items():
      function = getattr(lib, name)
      function.restype = result
      function.argtypes = args
    return lib
  except (OSError, AttributeError):
    return None


def run(graph_bytes, output_names):
  """Run all fetches in a fresh session, returning numeric NumPy outputs.

  Decline unsupported libraries, malformed graphs or unsupported output types
  so the caller can replay the original Python preprocessing path and errors.
  Every C allocation is owned by this call and released synchronously.
  """
  lib = _library()
  if lib is None:
    return None
  status = graph = buffer = import_options = session_options = session = None
  tensors = None
  try:
    if not isinstance(graph_bytes, bytes) or not isinstance(output_names, dict):
      return None
    status = lib.TF_NewStatus()
    graph = lib.TF_NewGraph()
    buffer = lib.TF_NewBufferFromString(graph_bytes, len(graph_bytes))
    import_options = lib.TF_NewImportGraphDefOptions()
    lib.TF_GraphImportGraphDef(graph, buffer, import_options, status)
    if lib.TF_GetCode(status):
      return None
    fetches = []
    for name in output_names.values():
      operation_name, index = name.rsplit(':', 1)
      index = int(index)
      operation = lib.TF_GraphOperationByName(graph, operation_name.encode())
      if not operation or not 0 <= index < lib.TF_OperationNumOutputs(operation):
        return None
      fetch = _Output(operation, index)
      if lib.TF_OperationOutputType(fetch) not in (*_DTYPES, 7):
        return None
      fetches.append(fetch)
    session_options = lib.TF_NewSessionOptions()
    lib.TF_SetConfig(session_options, _SESSION_CONFIG, len(_SESSION_CONFIG),
                     status)
    if lib.TF_GetCode(status):
      return None
    session = lib.TF_NewSession(graph, session_options, status)
    if not session or lib.TF_GetCode(status):
      return None
    outputs = (_Output * len(fetches))(*fetches)
    tensors = (ctypes.c_void_p * len(fetches))()
    lib.TF_SessionRun(
        session, None, None, None, 0, outputs, tensors, len(fetches),
        None, 0, None, status,
    )
    if lib.TF_GetCode(status):
      return None
    result = {}
    for key, tensor in zip(output_names, tensors):
      dtype_code = lib.TF_TensorType(tensor)
      if dtype_code == 7:
        continue  # Object/string outputs were still computed and fetched.
      dtype = np.dtype(_DTYPES[dtype_code])
      shape = tuple(lib.TF_Dim(tensor, i)
                    for i in range(lib.TF_NumDims(tensor)))
      size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
      if size != lib.TF_TensorByteSize(tensor):
        return None
      array = np.frombuffer(
          ctypes.string_at(lib.TF_TensorData(tensor), size), dtype=dtype
      ).copy().reshape(shape)
      result[key] = array if shape else array[()]
    return result
  except (OSError, ValueError, TypeError, KeyError, AttributeError):
    return None
  finally:
    if tensors is not None:
      for tensor in tensors:
        if tensor:
          lib.TF_DeleteTensor(tensor)
    if session:
      lib.TF_CloseSession(session, status)
      lib.TF_DeleteSession(session, status)
    if session_options:
      lib.TF_DeleteSessionOptions(session_options)
    if import_options:
      lib.TF_DeleteImportGraphDefOptions(import_options)
    if buffer:
      lib.TF_DeleteBuffer(buffer)
    if graph:
      lib.TF_DeleteGraph(graph)
    if status:
      lib.TF_DeleteStatus(status)


def release_host_memory():
  """Return freeable allocator-held pages to the OS (best-effort, no-op safe).

  Frees made by Python/numpy often stay mapped inside the allocator's arenas,
  so the process high-water mark (ru_maxrss) keeps blocks that are no longer
  live. Trimming after large transient allocations lets later stages reuse or
  skip that space, lowering the lifetime peak. Pure host-memory hygiene: no
  effect on any computed value. Returns True if a trim ran.
  """
  try:
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.malloc_trim(0)
    return True
  except (OSError, AttributeError):
    return False


def run_isolated(cache_path):
  """Run a cached feature graph in a child process, returning numeric outputs.

  The child re-parses the envelope and executes the identical graph bytes with
  the identical native library and session configuration, so outputs are
  bit-identical to run(); only the allocator high-water mark changes (it is
  charged to the child and released on child exit instead of raising this
  process's lifetime peak RSS). Returns None on any failure so the caller can
  fall back to the in-process path.
  """
  import os
  import pickle
  import subprocess
  import sys
  import tempfile
  try:
    fd, out_path = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)
    try:
      proc = subprocess.run(
          [sys.executable, '-m', 'alphafold.model.tf.graph_runtime_worker',
           str(cache_path), out_path],
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=900)
      if proc.returncode != 0:
        return None
      with open(out_path, 'rb') as f:
        return pickle.load(f)
    except (OSError, subprocess.SubprocessError, pickle.UnpicklingError,
            EOFError):
      return None
  finally:
    try:
      os.unlink(out_path)
    except OSError:
      pass
