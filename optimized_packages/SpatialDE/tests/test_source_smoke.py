"""Small software fixtures only; no synthetic biological benchmark/results."""
from pathlib import Path

import numpy as np
import pytest

import SpatialDE
from SpatialDE import base


def test_source_import_and_notice():
    package = Path(SpatialDE.__file__).resolve().parent
    assert package.name == "SpatialDE"
    assert (package / "LICENSE.scipy.txt").is_file()
    assert callable(SpatialDE.run)


def test_se_kernel_unchanged_clipping():
    coordinates = np.array([[0.], [1.], [2.]])
    distance2 = np.maximum((coordinates - coordinates.T) ** 2, 1e-12)
    expected = np.exp(-distance2 / (2 * 1.7 ** 2))
    np.testing.assert_array_equal(base.SE_kernel(coordinates, 1.7), expected)


@pytest.mark.parametrize("function", [base.mu_hat, base.LL])
def test_default_variance_matches_explicit_ones(function):
    y = np.array([.4, -.8, 1.2, .1])
    one = np.array([1.7, -.3, .2, .6])
    spectrum = np.array([.1, .4, 1.2, 2.3])
    implicit = function(.7, y, one, spectrum, 4)
    explicit = function(.7, y, one, spectrum, 4, np.ones(4))
    np.testing.assert_allclose(implicit, explicit, rtol=1e-12, atol=1e-12)


def test_variance_objective_and_optimizer_are_finite():
    y = np.array([.4, -.8, 1.2, .1])
    one = np.array([1.7, -.3, .2, .6])
    spectrum = np.array([.1, .4, 1.2, 2.3])
    np.testing.assert_allclose(
        base.s2_t_hat(.7, y, spectrum, 4),
        base.s2_t_hat(.7, y, spectrum, 4, np.ones(4)), rtol=1e-12, atol=1e-12)
    objective = base.make_objective(y, one, spectrum, 4)
    np.testing.assert_allclose(objective(.7), -base.LL(np.exp(.7), y, one, spectrum, 4),
                               rtol=1e-12, atol=1e-12)
    fit = base.lbfgsb_max_LL(y, one, spectrum, 4)
    # Public helper includes the fifth, curvature-derived log-delta uncertainty.
    assert len(fit) == 5 and np.isfinite(fit).all()
