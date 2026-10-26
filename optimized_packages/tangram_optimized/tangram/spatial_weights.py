"""
Function to compute spatial weights for the neighborhood extensions.
"""

def spatial_weights(adata_sp, standardized, self_inclusion):
    import numpy as np

    if not set(["spatial_connectivities", "spatial_distances"]).issubset(set(adata_sp.obsp.keys())):
        raise ValueError("Missing spatial neighborhood parameters. Run `pp_adatas()` with the spatial information stored in `spatial` in `adata_sp.obsm`.")

    if standardized:
        # Row-normalize the spatial distances matrix (standardized weights)
        # This replaces the libpysal detour with direct numpy operations
        g = adata_sp.obsp['spatial_distances'].toarray().astype(np.float64)
        row_sums = g.sum(axis=1, keepdims=True)
        # Avoid division by zero for isolated spots
        row_sums[row_sums == 0] = 1.0
        spatial_weights = np.asmatrix(g / row_sums)
    else: # binary
        spatial_weights = adata_sp.obsp['spatial_connectivities'].todense()
    if self_inclusion:
        spatial_weights = spatial_weights + np.eye(spatial_weights.shape[0])

    return spatial_weights
