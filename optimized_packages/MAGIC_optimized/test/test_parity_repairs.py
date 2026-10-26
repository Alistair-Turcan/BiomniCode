"""Numerical and behavioral regressions. Random arrays below are unit-test fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse, spatial
from sklearn.neighbors import NearestNeighbors

import magic
import graphtools


@pytest.fixture
def fixture():
    # Numerical test fixture, not simulated biological observations.
    a = np.random.RandomState(42).uniform(size=(80, 20))
    a[a < 0.85] = 0
    return a


def operator(**kwargs):
    defaults = dict(n_pca=10, knn=3, decay=1, t=3, solver="exact",
                    random_state=42, verbose=0, n_jobs=1)
    defaults.update(kwargs)
    return magic.MAGIC(**defaults)


def test_import_does_not_monkeypatch_sklearn():
    script = """import numpy as np
import sklearn.utils.extmath as e
a=np.random.RandomState(42).normal(size=(80,25))
rf,svd=e._randomized_range_finder,e._randomized_svd
before=e.randomized_svd(a,n_components=8,random_state=42)
import magic, graphtools
assert rf is e._randomized_range_finder and svd is e._randomized_svd
after=e.randomized_svd(a,n_components=8,random_state=42)
for x,y in zip(before,after): np.testing.assert_array_equal(x,y)
"""
    subprocess.run([sys.executable, "-c", script], check=True, env=os.environ.copy())


def test_fixed_exact_retains_float64_precision(fixture):
    op = operator().fit(fixture)
    expected = fixture
    for _ in range(3):
        expected = op.diff_op @ expected
    actual = op.transform(genes="all_genes")
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_public_input_types_unchanged(fixture):
    op = operator().fit(fixture)
    assert op.X is fixture
    assert isinstance(op.graph.data, np.ndarray)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fast_procrustes_matches_scipy(dtype):
    a = np.random.RandomState(42).normal(size=(80, 20)).astype(dtype)
    b = np.random.RandomState(43).normal(size=(80, 20)).astype(dtype)
    error, _ = operator()._calculate_error(a, b)
    expected = spatial.procrustes(b, a)[2]
    np.testing.assert_allclose(error, expected, rtol=1e-7, atol=1e-12)


def test_constant_procrustes_raises():
    with pytest.raises(ValueError, match="unique points"):
        operator()._calculate_error(np.ones((10, 3)), np.ones((10, 3)))


def test_identical_procrustes_is_nonnegative():
    a = np.random.RandomState(42).normal(size=(80, 20))
    error, _ = operator()._calculate_error(a, a)
    assert error >= 0
    np.testing.assert_allclose(error, spatial.procrustes(a, a)[2], atol=1e-12)


def test_sparse_no_pca_above_old_faiss_boundary():
    a = sparse.csr_matrix(np.arange(5001, dtype=float).reshape(-1, 1))
    g = graphtools.Graph(a, n_pca=None, knn=2, knn_max=2, decay=None, verbose=0)
    assert isinstance(g.knn_tree, NearestNeighbors)
    d, idx = g.knn_tree.radius_neighbors(a[:2], radius=0.5)
    np.testing.assert_array_equal(idx[0], [0])
    np.testing.assert_array_equal(idx[1], [1])


def test_ragged_neighbors_never_include_padding():
    import graphtools.graphs as graphs
    old = graphs.NUMBA_AVAILABLE
    graphs.NUMBA_AVAILABLE = False
    try:
        K = graphs._build_csr_from_neighbors([np.array([], dtype=int), np.array([1])],
                                            [np.array([]), np.array([0.])],
                                            1., 1., 0., (2, 2))
    finally:
        graphs.NUMBA_AVAILABLE = old
    assert np.all(K.indices >= 0)
    np.testing.assert_array_equal(K.toarray(), [[0, 0], [0, 1]])


@pytest.mark.parametrize("form", ["dense", "dataframe", "anndata"])
def test_sparse_output_toggle_and_labels(fixture, form):
    if form == "dataframe":
        X = pd.DataFrame(fixture, columns=[f"Gene_{i}" for i in range(20)])
    elif form == "anndata":
        import anndata
        X = anndata.AnnData(fixture)
        X.var_names = [f"Gene_{i}" for i in range(20)]
    else:
        X = fixture
    op = operator().fit(X)
    expected = op.transform(genes="all_genes")
    op.set_params(sparse_output=True)
    actual = op.transform(genes="all_genes")
    if form == "anndata":
        assert sparse.issparse(actual.X)
        assert list(actual.var_names) == list(X.var_names)
        expected, actual = expected.X, actual.X.toarray()
    elif form == "dataframe":
        assert list(actual.columns) == list(X.columns)
        assert all(isinstance(dt, pd.SparseDtype) for dt in actual.dtypes)
        expected, actual = expected.to_numpy(), actual.sparse.to_dense().to_numpy()
    else:
        assert sparse.isspmatrix_csr(actual)
        actual = actual.toarray()
    np.testing.assert_array_equal(actual, expected)


def test_parameter_roundtrip(fixture):
    from sklearn.base import clone
    op = operator(sparse_output=True)
    copied = clone(op)
    assert copied.get_params() == op.get_params()
    assert sparse.issparse(copied.fit_transform(fixture, genes="all_genes"))
