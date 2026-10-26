from ._version import __version__
from importlib import import_module as _import_module
from . import mapping_utils as _mapping_utils
from . import utils as _utils

# Preserve the original public exports, but ordinary mapping need not import
# Scanpy's filtering/plotting stack. Attribute access returns the actual objects.
_utility_exports = ("LeaveOneOut", "KFold", "auc")
for _module in (_mapping_utils, _utils):
    for _name in _module.__all__:
        if _name != "sc" and _name not in _utility_exports:
            globals()[_name] = getattr(_module, _name)

_plot_exports = (
    "plt", "sns", "entropy", "mpl", "GridSpec", "q_value",
    "plot_training_scores", "plot_gene_sparsity", "ordered_predictions",
    "convert_adata_array", "construct_obs_plot", "plot_cell_annotation_sc",
    "plot_cell_annotation", "plot_genes_sc", "plot_genes", "quick_plot_gene",
    "plot_annotation_entropy", "plot_test_scores", "plot_auc", "mapping_colors",
)

from .mapping_parameter_tuning import mapping_hyperparameter_tuning

__all__ = ([name for name in globals() if not name.startswith("_")]
           + ["sc", "plot_utils", *_utility_exports, *_plot_exports])


def __getattr__(name):
    if name == "sc":
        value = _import_module("scanpy")
    elif name == "plot_utils":
        value = _import_module(".plot_utils", __name__)
    elif name in _plot_exports:
        value = getattr(_import_module(".plot_utils", __name__), name)
    elif name in _utility_exports:
        value = getattr(_utils, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
