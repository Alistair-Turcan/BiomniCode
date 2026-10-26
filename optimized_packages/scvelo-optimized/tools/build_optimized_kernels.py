"""Rebuild the two shipped fixed-signature AOT modules from this checkout."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tools = root / "scvelo" / "tools"
    modules = ("_em_kernels_aot_np", "_em_kernels_aot_py")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"

    def libraries():
        return sorted({p for module in modules for pattern in ("*.so", "*.pyd")
                       for p in tools.glob(module + pattern)})

    # Build scripts need JIT dispatchers' .py_func, not the AOT-facing wrappers.
    with tempfile.TemporaryDirectory(prefix="scvelo-aot-") as backup:
        saved = []
        for path in libraries():
            target = Path(backup) / path.name
            shutil.move(str(path), str(target))
            saved.append((target, path))
        try:
            for module in modules:
                subprocess.run(
                    [sys.executable, str(tools / (module + "_build.py"))],
                    cwd=str(tools), env=env, check=True,
                )
                if not any(tools.glob(module + "*.so")) and not any(
                    tools.glob(module + "*.pyd")
                ):
                    raise RuntimeError("Compiler produced no library for " + module)
        except BaseException:
            for path in libraries():
                path.unlink()
            for source, destination in saved:
                shutil.move(str(source), str(destination))
            raise
    print("Built both AOT kernel modules. Validate them on this environment.")


if __name__ == "__main__":
    main()
