"""Optional native-vector loop, with a portable ordinary-Torch fallback.

The shared object is an ordinary rebuildable compiler cache outside the source
tree. It never contains scientific inputs. Compilation is attempted only for
the pinned Torch CPU ABI and supported instruction set.
"""
import ctypes
import hashlib
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import torch

_loaded = None
_library = None
_moment_function = None


def _load():
    global _loaded, _library, _moment_function
    if _loaded is not None:
        return _loaded
    _loaded = False
    compiler = shutil.which("c++")
    if (compiler is None or platform.system() != "Linux"
            or torch.__version__.split("+")[0] != "2.7.1"
            or torch.backends.cpu.get_cpu_capability() != "AVX512"):
        return False
    source = Path(__file__).with_suffix(".cpp")
    torch_root = Path(torch.__file__).parent
    flags = [
        "-std=c++17", "-O3", "-shared", "-fPIC", "-fopenmp",
        "-ffp-contract=off", "-DCPU_CAPABILITY_AVX512=1",
        "-DCPU_CAPABILITY=AVX512",
        "-mavx512f", "-mavx512dq", "-mavx512bw", "-mavx512vl", "-mavx2", "-mfma", "-mf16c",
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        "-I" + str(torch_root / "include"), "-L" + str(torch_root / "lib"),
        "-Wl,-rpath," + str(torch_root / "lib"),
    ]
    identity = hashlib.sha256(
        source.read_bytes() + repr((flags, compiler, torch.__version__)).encode()).hexdigest()
    cache = Path(tempfile.gettempdir()) / "tangram-native-update" / identity
    try:
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / "update.so"
        if not binary.is_file():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                output = Path(temporary) / "update.so"
                subprocess.run(
                    [compiler, *flags, str(source), "-ltorch_cpu", "-lc10", "-o", str(output)],
                    check=True, capture_output=True, timeout=120)
                output.replace(binary)
        _library = ctypes.CDLL(str(binary))
        function = _library.tangram_cpu_update
        function.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
            ctypes.c_float, ctypes.c_float, ctypes.c_int,
        ]
        function.restype = None
        _moment_function = _library.tangram_cpu_moments
        _moment_function.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
        _moment_function.restype = None
        _loaded = function
    except (OSError, subprocess.SubprocessError):
        _loaded = False
    return _loaded


def _moments(mean, square, gradient):
    if not all(type(t) is torch.Tensor and t.device.type == "cpu"
               and t.dtype == torch.float32 and t.is_contiguous() and t.shape == mean.shape
               for t in (mean, square, gradient)):
        return False
    if not _load():
        return False
    _moment_function(mean.data_ptr(), square.data_ptr(), gradient.data_ptr(),
                     mean.numel(), torch.get_num_threads())
    torch.autograd.graph.increment_version(mean)
    torch.autograd.graph.increment_version(square)
    torch.autograd.graph.increment_version(square)
    return True


def _update(parameter, mean, square_root, correction, value):
    if (type(parameter) is not torch.Tensor
            or parameter.device.type != "cpu" or parameter.dtype != torch.float32
            or not all(t.is_contiguous() and t.dtype == parameter.dtype
                       and t.device == parameter.device and t.shape == parameter.shape
                       for t in (parameter, mean, square_root))):
        return False
    function = _load()
    if not function:
        return False
    function(parameter.data_ptr(), mean.data_ptr(), square_root.data_ptr(), parameter.numel(),
             correction, value, torch.get_num_threads())
    torch.autograd.graph.increment_version(parameter)
    return True
