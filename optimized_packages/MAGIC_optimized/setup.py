"""Install the corrected vendored graphtools as its own normal distribution."""
from pathlib import Path
from setuptools import setup, Extension

ROOT = Path(__file__).parent
version = (ROOT / "graphtools/version.py").read_text().split('"')[1]
setup(
    name="graphtools",
    version=version,
    description="graphtools with audited MAGIC parity-preserving optimizations",
    packages=["graphtools"],
    license="GNU General Public License Version 2",
    python_requires=">=3.8",
    install_requires=["numpy>=1.14.0", "scipy>=1.1.0", "pygsp>=0.5.1",
                      "scikit-learn>=0.20.0", "future", "tasklogger>=1.0", "Deprecated"],
    extras_require={"numba": ["numba>=0.50.0"]},
    ext_modules=[Extension(
        "_magic_csr_native", ["native/magic_csr_native.c"], optional=True,
        extra_compile_args=["-O3", "-std=c99", "-march=native",
                            "-fno-fast-math", "-ffp-contract=off"],
    )],
)
