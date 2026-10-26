# -*- coding: utf-8 -*-

import logging
from ctypes import c_uint32
from math import ceil
from multiprocessing import Array, Process, cpu_count
from multiprocessing.sharedctypes import RawArray
from operator import attrgetter, mul
from typing import Sequence, Type

import numpy as np
import pandas as pd
from numba import njit, prange
from boltons.iterutils import chunked
from ctxcore.genesig import GeneSignature
from ctxcore.recovery import enrichment4cells, auc2d, derive_rank_cutoff
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
# To reduce the memory footprint of a ranking matrix we use unsigned 32bit integers which provides a range from 0
# through 4,294,967,295. This should be sufficient even for region-based approaches.
DTYPE = "uint32"
DTYPE_C = c_uint32


@njit(parallel=True, cache=True)
def _sparse_rankings_numba(values, positions):
    n_cells, n_genes = values.shape
    rankings = np.empty((n_cells, n_genes), dtype=np.uint32)
    for cell in prange(n_cells):
        row = values[cell, positions]
        nonzero = np.flatnonzero(row != 0)
        order = nonzero[np.argsort(-row[nonzero], kind="mergesort")]
        n_positive = np.count_nonzero(row > 0)
        zeros = np.flatnonzero(row == 0)
        rank = 0
        for i in range(n_positive):
            rankings[cell, order[i]] = rank
            rank += 1
        for i in range(len(zeros)):
            rankings[cell, zeros[i]] = rank
            rank += 1
        for i in range(n_positive, len(order)):
            rankings[cell, order[i]] = rank
            rank += 1
    return rankings


@njit(cache=True, parallel=True)
def _indexed_auc_parallel(rankings, gene_indices, weights, rank_cutoff, max_auc):
    """Score selected columns without materializing their entire rank matrix."""
    n_features = rankings.shape[0]
    n_genes = len(gene_indices)
    aucs = np.empty(n_features, dtype=np.float64)
    for row_idx in prange(n_features):
        x_tmp = np.empty(n_genes + 1, dtype=np.int64)
        y_tmp = np.empty(n_genes + 1, dtype=np.float64)
        n_filtered = 0
        for j in range(n_genes):
            rank = rankings[row_idx, gene_indices[j]]
            if rank < rank_cutoff:
                x_tmp[n_filtered] = rank
                y_tmp[n_filtered] = weights[j]
                n_filtered += 1
        if n_filtered == 0:
            aucs[row_idx] = 0.0
            continue
        sort_idx = np.argsort(x_tmp[:n_filtered])
        total = 0.0
        prev_x = 0
        auc_sum = 0.0
        for i in range(n_filtered):
            curr_x = x_tmp[sort_idx[i]]
            if i > 0:
                auc_sum += (curr_x - prev_x) * total
            total += y_tmp[sort_idx[i]]
            prev_x = curr_x
        auc_sum += (rank_cutoff - prev_x) * total
        aucs[row_idx] = auc_sum / max_auc
    return aucs


@njit(cache=True, parallel=True)
def _cutoff_gene_index(rankings, rank_cutoff):
    top = np.full((rankings.shape[0], rank_cutoff), -1, dtype=np.int32)
    valid = np.ones(rankings.shape[0], dtype=np.bool_)
    for row in prange(rankings.shape[0]):
        found = 0
        for gene in range(rankings.shape[1]):
            rank = rankings[row, gene]
            if rank < rank_cutoff:
                if top[row, rank] != -1:
                    valid[row] = False
                else:
                    top[row, rank] = gene
                    found += 1
        if found != rank_cutoff:
            valid[row] = False
    return top, valid


@njit(cache=True, parallel=True)
def _cutoff_index_auc(top, members, weights, maxauc):
    aucs = np.empty(top.shape[0], dtype=np.float64)
    rank_cutoff = top.shape[1]
    for row in prange(top.shape[0]):
        total, auc_sum = 0.0, 0.0
        previous = 0
        found = False
        for rank in range(rank_cutoff):
            gene = top[row, rank]
            if members[gene]:
                if found:
                    auc_sum += (rank - previous) * total
                total += weights[gene]
                previous = rank
                found = True
        if found:
            auc_sum += (rank_cutoff - previous) * total
            aucs[row] = auc_sum / maxauc
        else:
            aucs[row] = 0.0
    return aucs


def create_rankings(ex_mtx: pd.DataFrame, seed=None) -> pd.DataFrame:
    """
    Create a whole genome rankings dataframe from a single cell expression profile dataframe.

    :param ex_mtx: The expression profile matrix. The rows should correspond to different cells, the columns to different
        genes (n_cells x n_genes).
    :return: A genome rankings dataframe (n_cells x n_genes).
    """
    # Do a shuffle would be nice for exactly similar behaviour as R implementation.
    # 1. Ranks are assigned in the range of 1 to n, therefore we need to subtract 1.
    # 2. In case of a tie the 'first' method is used, i.e. we keep the order in the original array. The remove any
    #    bias we shuffle the dataframe before ranking it. This introduces a performance penalty!
    # 3. Genes are ranked according to gene expression in descending order, i.e. from highly expressed (0) to low expression (n).
    # 3. NAs should be given the highest rank numbers. Documentation is bad, so tested implementation via code snippet:
    #
    #    import pandas as pd
    #    import numpy as np
    #    df = pd.DataFrame(data=[4, 1, 3, np.nan, 2, 3], columns=['values'])
    #    # Run below statement multiple times to see effect of shuffling in case of a tie.
    #    df.sample(frac=1.0, replace=False).rank(ascending=False, method='first', na_option='bottom').sort_index() - 1
    #
    # Sample positions using pandas' own RNG handling, exactly as the upstream
    # shared column shuffle. Sampling only positions avoids copying expression.
    positions = pd.Series(np.arange(ex_mtx.shape[1])).sample(
        frac=1.0, replace=False, random_state=seed
    ).to_numpy()
    values = ex_mtx.to_numpy()
    columns = ex_mtx.columns.take(positions)
    if values.dtype.kind != "f":
        # pandas handles heterogeneous/nullable types and large integers without
        # our float coercion changing comparisons or unsigned negation overflowing.
        return ex_mtx.iloc[:, positions].rank(
            axis=1, ascending=False, method="first", na_option="bottom"
        ).astype(DTYPE) - 1
    n_cells, n_genes = values.shape
    if values.dtype in (np.dtype("float32"), np.dtype("float64")) and values.dtype.isnative:
        rankings = _sparse_rankings_numba(values, positions)
        result = pd.DataFrame(rankings, index=ex_mtx.index, columns=columns)
        result.attrs = ex_mtx.attrs.copy()
        return result
    # uint32 is the public upstream contract, even for small gene panels.
    rankings = np.empty((n_cells, n_genes), dtype=DTYPE)
    rank_values = np.arange(n_genes, dtype=DTYPE)
    for i in range(n_cells):
        row = values[i, positions]
        nonzero = np.flatnonzero(row != 0)
        # Stable descending sort preserves shuffled order for ties; NaNs sort
        # last. Zeros belong between positive and negative expression, not last.
        order = nonzero[np.argsort(-row[nonzero], kind="stable")]
        n_positive = np.count_nonzero(row > 0)
        zero = np.flatnonzero(row == 0)
        order = np.concatenate((order[:n_positive], zero, order[n_positive:]))
        rankings[i, order] = rank_values
    result = pd.DataFrame(rankings, index=ex_mtx.index, columns=columns)
    result.attrs = ex_mtx.attrs.copy()
    return result


def derive_auc_threshold(ex_mtx: pd.DataFrame) -> pd.DataFrame:
    """
    Derive AUC thresholds for an expression matrix.

    It is important to check that most cells have a substantial fraction of expressed/detected genes in the calculation of
    the AUC.

    :param ex_mtx: The expression profile matrix. The rows should correspond to different cells, the columns to different
        genes (n_cells x n_genes).
    :return: A dataframe with AUC threshold for different quantiles over the number cells: a fraction of 0.01 designates
        that when using this value as the AUC threshold for 99% of the cells all ranked genes used for AUC calculation will
        have had a detected expression in the single-cell experiment.
    """
    return (
        pd.Series(np.count_nonzero(ex_mtx, axis=1)).quantile(
            [0.01, 0.05, 0.10, 0.50, 1]
        )
        / ex_mtx.shape[1]
    )


enrichment = enrichment4cells


def _enrichment(
    shared_ro_memory_array, modules, genes, cells, auc_threshold, auc_mtx, offset
):
    # The rankings dataframe is properly reconstructed (checked this).
    df_rnk = pd.DataFrame(
        data=np.frombuffer(shared_ro_memory_array, dtype=DTYPE).reshape(
            len(cells), len(genes)
        ),
        columns=genes,
        index=cells,
    )
    # To avoid additional memory burden de resulting AUCs are immediately stored in the output sync. array.
    result_mtx = np.frombuffer(auc_mtx.get_obj(), dtype="d")
    inc = len(cells)
    for idx, module in enumerate(modules):
        result_mtx[
            offset + (idx * inc) : offset + ((idx + 1) * inc)
        ] = enrichment4cells(df_rnk, module, auc_threshold).values.ravel(order="C")


def aucell4r(
    df_rnk: pd.DataFrame,
    signatures: Sequence[Type[GeneSignature]],
    auc_threshold: float = 0.05,
    noweights: bool = False,
    normalize: bool = False,
    num_workers: int = cpu_count(),
) -> pd.DataFrame:
    """
    Calculate enrichment of gene signatures for single cells.

    :param df_rnk: The rank matrix (n_cells x n_genes).
    :param signatures: The gene signatures or regulons.
    :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    :param noweights: Should the weights of the genes part of a signature be used in calculation of enrichment?
    :param normalize: Normalize the AUC values to a maximum of 1.0 per regulon.
    :param num_workers: The number of cores to use.
    :return: A dataframe with the AUCs (n_cells x n_modules).
    """
    if num_workers == 1:
        # The upstream concat/unstack path defines unusual-label, duplicate, and
        # empty-input behavior. Preserve it rather than silently collapsing keys.
        signature_names = pd.Index([s.name for s in signatures])
        if (not len(signatures) or not len(df_rnk.index)
                or not df_rnk.index.is_unique or not df_rnk.columns.is_unique
                or not signature_names.is_unique
                or isinstance(df_rnk.index, pd.MultiIndex)
                or df_rnk.index.hasnans
                or df_rnk.index.inferred_type.startswith("mixed")):
            aucs_df = pd.concat([
                enrichment4cells(
                    df_rnk, module.noweights() if noweights else module,
                    auc_threshold=auc_threshold,
                ) for module in signatures
            ]).unstack("Regulon")
            aucs_df.columns = aucs_df.columns.droplevel(0)
            return aucs_df / aucs_df.max(axis=0) if normalize else aucs_df
        # Optimized single-worker path: compute AUCs directly as numpy arrays
        # and construct the final DataFrame once, instead of creating one DataFrame
        # per regulon (via enrichment4cells) then concat + unstack.
        # Uses numpy array indexing instead of DataFrame.iloc for column selection.
        total_genes = len(df_rnk.columns)
        cells = df_rnk.index.values
        n_cells = len(cells)
        n_sigs = len(signatures)
        col_names = list(df_rnk.columns)  # list for fast integer indexing (22x faster than ArrowStringArray)
        col_to_idx = {name: i for i, name in enumerate(col_names)}
        rnk_values = df_rnk.values
        rank_cutoff = None  # Upstream validates only signatures meeting coverage.
        auc_matrix = np.empty((n_sigs, n_cells), dtype=np.float64)
        top_genes = None
        top_checked = False
        top_eligible = (
            rnk_values.dtype == np.dtype("uint32") and n_cells >= 256
            and n_sigs >= 8 and total_genes <= np.iinfo(np.int32).max
        )
        for idx, module in enumerate(signatures):
            module = module.noweights() if noweights else module
            gene_indices = np.array(
                sorted(col_to_idx[g] for g in module.genes if g in col_to_idx),
                dtype=np.int64,
            )
            if len(gene_indices) == 0 or (float(len(gene_indices)) / float(len(module))) < 0.80:
                LOGGER.warning(
                    f"Less than 80% of the genes in {module.name} are present in the "
                    "expression matrix."
                )
                auc_matrix[idx] = 0.0
            else:
                if rank_cutoff is None:
                    rank_cutoff = derive_rank_cutoff(auc_threshold, total_genes)
                gene2weight = module.gene2weight
                weights = np.array(
                    [gene2weight.get(col_names[gi], 1.0) for gi in gene_indices]
                )
                y_max = weights.sum()
                maxauc = float((rank_cutoff + 1) * y_max)
                assert maxauc > 0
                if top_eligible and not top_checked:
                    top, valid = _cutoff_gene_index(rnk_values, rank_cutoff)
                    top_checked = True
                    if valid.all():
                        top_genes = top
                supported_weights = weights.dtype.kind in "ifu" and 4 <= weights.dtype.itemsize <= 8
                # Dispatch small inputs to the retained serial compiled kernel.
                if top_genes is not None and supported_weights:
                    members = np.zeros(total_genes, dtype=np.bool_)
                    dense_weights = np.zeros(total_genes, dtype=np.float64)
                    members[gene_indices] = True
                    dense_weights[gene_indices] = weights
                    auc_matrix[idx] = _cutoff_index_auc(
                        top_genes, members, dense_weights, maxauc
                    )
                elif rnk_values.dtype == np.dtype("uint32") and n_cells >= 256:
                    auc_matrix[idx] = _indexed_auc_parallel(
                        rnk_values, gene_indices, weights, rank_cutoff, maxauc
                    )
                else:
                    rnk_subset = rnk_values[:, gene_indices]
                    auc_matrix[idx] = auc2d(rnk_subset, weights, rank_cutoff, maxauc)
        aucs_df = pd.DataFrame(
            data=auc_matrix.T,
            columns=pd.Index(
                data=list(map(attrgetter("name"), signatures)), name="Regulon"
            ),
            index=pd.Index(data=cells, name="Cell"),
            copy=False,
        ).sort_index(axis=0).sort_index(axis=1)
    else:
        # Decompose the rankings dataframe: the index and columns are shared with the child processes via pickling.
        genes = df_rnk.columns.values
        cells = df_rnk.index.values
        # The actual rankings are shared directly. This is possible because during a fork from a parent process the child
        # process inherits the memory of the parent process. A RawArray is used instead of a synchronize Array because
        # these rankings are read-only.
        shared_ro_memory_array = RawArray(DTYPE_C, mul(*df_rnk.shape))
        array = np.frombuffer(shared_ro_memory_array, dtype=DTYPE)
        # Copy the contents of df_rank into this shared memory block using row-major ordering.
        array[:] = df_rnk.values.ravel(order="C")

        # The resulting AUCs are returned via a synchronize array.
        auc_mtx = Array("d", len(cells) * len(signatures))  # Double precision floats.

        # Convert the modules to modules with uniform weights if necessary.
        if noweights:
            signatures = list(map(lambda m: m.noweights(), signatures))

        # Do the analysis in separate child processes.
        chunk_size = ceil(float(len(signatures)) / num_workers)
        processes = [
            Process(
                target=_enrichment,
                args=(
                    shared_ro_memory_array,
                    chunk,
                    genes,
                    cells,
                    auc_threshold,
                    auc_mtx,
                    (chunk_size * len(cells)) * idx,
                ),
            )
            for idx, chunk in enumerate(chunked(signatures, chunk_size))
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # Reconstitute the results array. Using C or row-major ordering.
        aucs = pd.DataFrame(
            data=np.ctypeslib.as_array(auc_mtx.get_obj()).reshape(
                len(signatures), len(cells)
            ),
            columns=pd.Index(data=cells, name="Cell"),
            index=pd.Index(
                data=list(map(attrgetter("name"), signatures)), name="Regulon"
            ),
        ).T
    if num_workers == 1:
        return aucs_df / aucs_df.max(axis=0) if normalize else aucs_df
    else:
        return aucs / aucs.max(axis=0) if normalize else aucs


def aucell(
    exp_mtx: pd.DataFrame,
    signatures: Sequence[Type[GeneSignature]],
    auc_threshold: float = 0.05,
    noweights: bool = False,
    normalize: bool = False,
    seed=None,
    num_workers: int = cpu_count(),
) -> pd.DataFrame:
    """
    Calculate enrichment of gene signatures for single cells.

    :param exp_mtx: The expression matrix (n_cells x n_genes).
    :param signatures: The gene signatures or regulons.
    :param auc_threshold: The fraction of the ranked genome to take into account for the calculation of the
        Area Under the recovery Curve.
    :param noweights: Should the weights of the genes part of a signature be used in calculation of enrichment?
    :param normalize: Normalize the AUC values to a maximum of 1.0 per regulon.
    :param num_workers: The number of cores to use.
    :return: A dataframe with the AUCs (n_cells x n_modules).
    """
    # Create rankings and free the expression matrix to reduce peak memory
    df_rnk = create_rankings(exp_mtx, seed)
    del exp_mtx
    return aucell4r(
        df_rnk,
        signatures,
        auc_threshold,
        noweights,
        normalize,
        num_workers,
    )
