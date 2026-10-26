"""Standalone tests for the speed-preserving numerical compatibility repairs."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from torch.nn.functional import cosine_similarity

import tangram as tg
from tangram.mapping_optimizer import Mapper, MapperConstrained, _cosine_cached, _unit_reference


@pytest.mark.parametrize("scale", [0., 1e-10, 1e-7, 1., 1e4])
@pytest.mark.parametrize("dim", [0, 1])
def test_cached_cosine_values_and_gradients(scale, dim):
    x = (torch.arange(1, 25, dtype=torch.float32).reshape(6, 4) * scale).requires_grad_()
    y = torch.flip(x.detach(), [0])
    a = cosine_similarity(x, y, dim=dim)
    ga, = torch.autograd.grad(a.sum(), x)
    b = _cosine_cached(x, _unit_reference(y, dim), dim)
    gb, = torch.autograd.grad(b.sum(), x)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ga, gb, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("constrained", [False, True])
def test_zero_density_is_finite(constrained):
    S = np.arange(1, 25, dtype="float32").reshape(6, 4)
    G = np.arange(1, 13, dtype="float32").reshape(3, 4)
    cls = MapperConstrained if constrained else Mapper
    model = cls(S, G, d=np.array([0., .3, .7]), lambda_d=1, random_state=42)
    loss = model._loss_fn(False)[0]
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.M.grad).all()


@pytest.mark.parametrize("cls", [Mapper, MapperConstrained])
def test_constructor_owns_numpy_inputs(cls):
    S, G = np.ones((6, 4), dtype="float32"), np.ones((3, 4), dtype="float32")
    kw = {"d": None} if cls is MapperConstrained else {}
    model = cls(S, G, **kw)
    S[:] = 0
    G[:] = 0
    assert torch.all(model.S == 1) and torch.all(model.G == 1)


def test_gene_matching_preserves_order_duplicates_and_first_match():
    result = tg.get_matched_genes(["z", "a", "z", "b"], ["b", "z", "a", "z"])
    assert result == ([3, 0, 1, 0], [0, 1, 2, 3], ["b", "z", "a", "z"])


def test_one_hot_preserves_label_types_order_and_index():
    labels = pd.Series([3, 1, 3], index=["c3", "c1", "c2"])
    actual = tg.one_hot_encoding(labels)
    assert list(actual.columns) == [3, 1]
    assert list(actual.index) == list(labels.index)
    np.testing.assert_array_equal(actual, [[1, 0], [0, 1], [1, 0]])


def test_sparse_sparsity_counts_values_not_storage():
    import anndata as ad
    X = sp.csr_matrix((np.array([1., -1., 0., 2.]),
                       np.array([0, 0, 1, 1]), np.array([0, 3, 4])), shape=(2, 2))
    data = ad.AnnData(X)
    tg.annotate_gene_sparsity(data)
    np.testing.assert_array_equal(data.var["sparsity"], [1., .5])


@pytest.mark.parametrize("form", ["dense", "csr", "csc"])
@pytest.mark.parametrize("dtype", ["int8", "bool", "float32", "float64"])
@pytest.mark.parametrize("scale", [False, True])
def test_cluster_aggregation_preserves_native_reduction(form, dtype, scale):
    import anndata as ad
    X = (np.arange(48).reshape(12, 4) + 30).astype(dtype)
    if form != "dense":
        X = {"csr": sp.csr_matrix, "csc": sp.csc_matrix}[form](X)
    data = ad.AnnData(X, obs=pd.DataFrame({"ct": ["b", "a"] * 6},
                                         index=[f"c{i}" for i in range(12)]))
    result = tg.adata_to_cluster_expression(data, "ct", scale=scale)
    assert result.X.dtype == np.dtype("float64")
    for i, label in enumerate(result.obs["ct"]):
        group = data[data.obs["ct"] == label].X
        expected = group.sum(0) if scale else group.mean(0)
        np.testing.assert_allclose(result.X[i], np.asarray(expected).reshape(-1),
                                   atol=1e-6, rtol=1e-5)


def test_stable_spatial_statistics_for_large_offsets():
    rng = np.random.default_rng(91)
    S, G = rng.uniform(1, 3, (12, 7)).astype("float32"), rng.uniform(1, 3, (8, 7)).astype("float32")
    W = rng.random((8, 8)).astype("float32")
    model = Mapper(S + 10000, G + 10000, lambda_geary=1, lambda_moran=1, spatial_weights=W)
    x, w = model.G_train, model.spatial_weights
    z = x - x.mean(0)
    moran = len(x) * z * (w @ z) / (z*z).sum(0)
    var = (z*z).sum(0) / (len(x)-1)
    geary = (w[:, :, None] * (x[:, None, :] - x[None, :, :])**2).sum((0, 1)) / (2*var)
    _, actual_moran, actual_geary = model._spatial_local_indicators(x)
    torch.testing.assert_close(actual_moran, moran, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual_geary, geary, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("form", ["dense", "csr"])
def test_cluster_aggregation_ignores_missing_labels(form):
    import anndata as ad
    X = np.arange(20, dtype="float32").reshape(5, 4)
    if form == "csr":
        X = sp.csr_matrix(X)
    data = ad.AnnData(X, obs=pd.DataFrame({"ct": ["a", None, "b", "a", None]},
                                         index=[f"c{i}" for i in range(5)]))
    result = tg.adata_to_cluster_expression(data, "ct", scale=False)
    for i, label in enumerate(result.obs["ct"]):
        expected = data[data.obs["ct"] == label].X.mean(0)
        np.testing.assert_allclose(result.X[i], np.asarray(expected).reshape(-1),
                                   atol=1e-6, rtol=1e-5)


def test_island_penalty_tie_subgradient_matches_maximum():
    model = Mapper(np.ones((6, 4)), np.ones((3, 4)),
                   lambda_ct_islands=1,
                   ct_encode=np.eye(2)[np.arange(6) % 2],
                   neighborhood_filter=np.array([[0., 1., 0.],
                                                 [0., 1., 0.],
                                                 [0., 1., 0.]]))
    with torch.no_grad():
        model.M.zero_()
    prob = torch.softmax(model.M, dim=1)
    ct = prob.T @ model.ct_encode
    reference = torch.maximum(ct-model.neighborhood_filter@ct, torch.zeros(1)).mean()
    expected, = torch.autograd.grad(reference, model.M)
    model._loss_fn(False)[0].backward()
    torch.testing.assert_close(model.M.grad, expected, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cpu:0"])
def test_history_schema_and_device_strings(device):
    model = Mapper(np.ones((6, 4)), np.ones((3, 4)), device=device, random_state=42)
    _, history = model.train(2, print_each=None)
    assert isinstance(history["total_loss"][0], np.ndarray)
    assert history["total_loss"][0].shape == ()
    assert isinstance(history["main_loss"][0], float)
    constrained = MapperConstrained(np.ones((6, 4)), np.ones((3, 4)), d=None, device=device)
    _, _, history = constrained.train(2, print_each=None)
    assert history["total_loss"][0].startswith("tensor(")
    assert "grad_fn=<AddBackward0>" in history["total_loss"][0]
