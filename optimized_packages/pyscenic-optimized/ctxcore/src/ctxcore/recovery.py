from __future__ import annotations

import logging
from itertools import repeat
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numba import jit, prange

if TYPE_CHECKING:
    from ctxcore.genesig import GeneSignature
    from ctxcore.rnkdb import RankingDatabase

__all__ = [
    "recovery",
    "aucs",
    "enrichment4features",
    "enrichment4cells",
    "leading_edge4row",
    "rcc2d_stats",
    "rcc2d_subset",
]


LOGGER = logging.getLogger(__name__)


def derive_rank_cutoff(
    auc_threshold: float, total_genes: int, rank_threshold: int | None = None
) -> int:
    """
    Get rank cutoff.

    :param auc_threshold: The fraction of the ranked genome to take into account for
        the calculation of the Area Under the recovery Curve.
    :param total_genes: The total number of genes ranked.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :return Rank cutoff.
    """
    if not rank_threshold:
        rank_threshold = total_genes - 1

    assert 0 < rank_threshold < total_genes, (
        f"Rank threshold must be an integer between 1 and {total_genes:d}."
    )
    assert 0.0 < auc_threshold <= 1.0, (
        "AUC threshold must be a fraction between 0.0 and 1.0."
    )

    # In the R implementation the cutoff is rounded.
    rank_cutoff = round(auc_threshold * total_genes)
    assert 0 < rank_cutoff <= rank_threshold, (
        f"An AUC threshold of {auc_threshold:f} corresponds to {rank_cutoff:d} top "
        f"ranked genes/regions in the database. Please increase the rank threshold "
        "or decrease the AUC threshold."
    )
    # Make sure we have exactly the same AUC values as the R-SCENIC pipeline.
    # In the latter the rank threshold is not included in AUC calculation.
    rank_cutoff -= 1
    return rank_cutoff


@jit(nopython=True, parallel=True, cache=True)
def _rcc2d_numba(rankings: np.ndarray, weights: np.ndarray, rank_threshold: int) -> np.ndarray:
    """
    Numba-compiled 2D recovery curve kernel. Replaces the Python loop with
    a single compiled pass over all features, avoiding per-row np.bincount
    and np.cumsum overhead. Uses parallel=True for multi-core execution.
    """
    n_features = rankings.shape[0]
    n_genes = rankings.shape[1]
    rccs = np.empty(shape=(n_features, rank_threshold), dtype=np.float64)
    for row_idx in prange(n_features):
        # Build weighted histogram (equivalent to bincount with weights)
        counts = np.zeros(rank_threshold, dtype=np.float64)
        for j in range(n_genes):
            r = rankings[row_idx, j]
            if r < rank_threshold:
                counts[r] += weights[j]
        # Cumulative sum
        cumsum = 0.0
        for r in range(rank_threshold):
            cumsum += counts[r]
            rccs[row_idx, r] = cumsum
    return rccs


@jit(nopython=True, parallel=True, cache=True)
def _rcc2d_with_auc_numba(
    rankings: np.ndarray, weights: np.ndarray, rank_threshold: int, rank_cutoff: int, maxauc: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fused recovery curve + AUC kernel. Computes RCCs and AUCs in a single
    pass, avoiding the separate rccs[:, :rank_cutoff].sum(axis=1) call.
    """
    n_features = rankings.shape[0]
    n_genes = rankings.shape[1]
    rccs = np.empty(shape=(n_features, rank_threshold), dtype=np.float64)
    aucs = np.empty(shape=(n_features,), dtype=np.float64)
    for row_idx in prange(n_features):
        counts = np.zeros(rank_threshold, dtype=np.float64)
        for j in range(n_genes):
            r = rankings[row_idx, j]
            if r < rank_threshold:
                counts[r] += weights[j]
        cumsum = 0.0
        auc_sum = 0.0
        for r in range(rank_threshold):
            cumsum += counts[r]
            rccs[row_idx, r] = cumsum
            if r < rank_cutoff:
                auc_sum += cumsum
        aucs[row_idx] = auc_sum / maxauc
    return rccs, aucs


@jit(nopython=True, cache=True)
def _rcc2d_stats_numba(rankings: np.ndarray, weights: np.ndarray, rank_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std of recovery curves without materializing the full RCC matrix.
    Serial (no parallel) to avoid race conditions on accumulation arrays.
    Saves ~90% memory vs full rcc2d + mean/std.
    """
    n_features = rankings.shape[0]
    n_genes = rankings.shape[1]
    sum_rcc = np.zeros(rank_threshold, dtype=np.float64)
    for row_idx in range(n_features):
        counts = np.zeros(rank_threshold, dtype=np.float64)
        for j in range(n_genes):
            r = rankings[row_idx, j]
            if r < rank_threshold:
                counts[r] += weights[j]
        cumsum = 0.0
        for r in range(rank_threshold):
            cumsum += counts[r]
            sum_rcc[r] += cumsum
    mean_rcc = sum_rcc / n_features
    # A second streaming pass matches the centered population variance used by
    # NumPy. E[x*x]-E[x]**2 catastrophically cancels for large/constant curves.
    sum_centered_sq = np.zeros(rank_threshold, dtype=np.float64)
    for row_idx in range(n_features):
        counts = np.zeros(rank_threshold, dtype=np.float64)
        for j in range(n_genes):
            r = rankings[row_idx, j]
            if r < rank_threshold:
                counts[r] += weights[j]
        cumsum = 0.0
        for r in range(rank_threshold):
            cumsum += counts[r]
            delta = cumsum - mean_rcc[r]
            sum_centered_sq[r] += delta * delta
    std_rcc = np.sqrt(sum_centered_sq / n_features)
    return mean_rcc, std_rcc


@jit(nopython=True, parallel=True, cache=True)
def _rcc2d_subset_numba(rankings: np.ndarray, weights: np.ndarray, rank_threshold: int, feature_indices: np.ndarray) -> np.ndarray:
    """
    Compute recovery curves for a subset of features (e.g. enriched features only).
    Parallelized with prange. Saves memory by only materializing RCCs for the subset.
    """
    n_subset = len(feature_indices)
    n_genes = rankings.shape[1]
    rccs = np.empty(shape=(n_subset, rank_threshold), dtype=np.float64)
    for idx in prange(n_subset):
        row_idx = feature_indices[idx]
        counts = np.zeros(rank_threshold, dtype=np.float64)
        for j in range(n_genes):
            r = rankings[row_idx, j]
            if r < rank_threshold:
                counts[r] += weights[j]
        cumsum = 0.0
        for r in range(rank_threshold):
            cumsum += counts[r]
            rccs[idx, r] = cumsum
    return rccs


def rcc2d_stats(rankings: np.ndarray, weights: np.ndarray, rank_threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std of recovery curves without materializing the full RCC matrix.

    :param rankings: The features rankings for a gene signature (n_features, n_genes).
    :param weights: The weights of these genes.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :return: A tuple of (mean_rcc, std_rcc), each of shape (rank_threshold,).
    """
    if not _kernel_inputs_supported(rankings, weights) or not len(rankings):
        full = rcc2d(rankings, weights, rank_threshold)
        return full.mean(axis=0), full.std(axis=0)
    return _rcc2d_stats_numba(rankings, weights, rank_threshold)


def rcc2d_subset(rankings: np.ndarray, weights: np.ndarray, rank_threshold: int, feature_indices: np.ndarray) -> np.ndarray:
    """
    Compute recovery curves for a subset of features.

    :param rankings: The features rankings for a gene signature (n_features, n_genes).
    :param weights: The weights of these genes.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :param feature_indices: Indices of features to compute RCCs for.
    :return: Recovery curves (n_subset, rank_threshold).
    """
    feature_indices = np.asarray(feature_indices)
    # NumPy indexing supplies validation, including negative-index handling.
    checked_indices = np.arange(len(rankings))[feature_indices]
    if not _kernel_inputs_supported(rankings, weights):
        return rcc2d(rankings, weights, rank_threshold)[checked_indices]
    return _rcc2d_subset_numba(rankings, weights, rank_threshold, checked_indices)


def rcc2d(rankings: np.ndarray, weights: np.ndarray, rank_threshold: int) -> np.ndarray:
    """
    Calculate recovery curves.

    :param rankings: The features rankings for a gene signature (n_features, n_genes).
    :param weights: The weights of these genes.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :return: Recovery curves (n_features, rank_threshold).
    """
    # Keep np.bincount's dtype/negative/short-histogram semantics for unusual
    # direct callers. The compiled path is safe for ordinary padded rankings.
    if (not _kernel_inputs_supported(rankings, weights)
            or rank_threshold < 1
            or rankings.shape[1] == 0
            or (rankings.shape[0] and
                np.any(rankings.max(axis=1) < rank_threshold - 1))):
        return _rcc2d_reference(rankings, weights, rank_threshold)
    return _rcc2d_numba(rankings, weights, rank_threshold)


def _kernel_inputs_supported(rankings, weights):
    return (
        isinstance(rankings, np.ndarray) and rankings.ndim == 2
        and rankings.dtype.kind in "iu" and rankings.dtype != np.dtype("uint64")
        and (rankings.dtype.kind == "u" or not rankings.size or rankings.min() >= 0)
        and isinstance(weights, np.ndarray) and weights.ndim == 1
        and len(weights) == rankings.shape[1]
        and weights.dtype.kind in "ifu" and weights.dtype.itemsize >= 4
    )


def _rcc2d_reference(rankings, weights, rank_threshold):
    """Unchanged upstream behavior for inputs outside the optimized contract."""
    rccs = np.empty((rankings.shape[0], rank_threshold))
    for row_idx in range(rankings.shape[0]):
        rccs[row_idx] = np.cumsum(
            np.bincount(rankings[row_idx], weights=weights)[:rank_threshold]
        )
    return rccs


def recovery(
    rnk: pd.DataFrame,
    total_genes: int,
    weights: np.ndarray,
    rank_threshold: int,
    auc_threshold: float,
    no_auc: bool = False,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate recovery curves and AUCs. This is the workhorse of the recovery algorithm.

    :param rnk: A dataframe containing the rank number of genes of interest.
        Columns correspond to genes.
    :param total_genes: The total number of genes ranked.
    :param weights: the weights associated with the selected genes.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :param auc_threshold: The fraction of the ranked genome to take into account for
        the calculation of the Area Under the recovery Curve.
    :param no_auc: Do not calculate AUCs.
    :return: A tuple of numpy arrays. The first array contains the recovery curves
        (n_features/n_cells x rank_threshold),
        the second array the AUC values (n_features/n_cells).
    """
    rank_cutoff = derive_rank_cutoff(auc_threshold, total_genes, rank_threshold)
    features, _genes, rankings = rnk.index.values, rnk.columns.values, rnk.values
    if not _kernel_inputs_supported(rankings, weights):
        # The sentinel is essential to upstream behavior for unsupported dtypes
        # and direct bincount validation; retain it only on this fallback path.
        padded_weights = np.insert(weights, len(weights), 0.0)
        padded_rankings = np.append(
            rankings, np.full((len(features), 1), total_genes), axis=1
        )
        rccs = _rcc2d_reference(padded_rankings, padded_weights, rank_threshold)
        if no_auc:
            return rccs, np.array([])
        maxauc = float((rank_cutoff + 1) * padded_weights.sum())
        assert maxauc > 0
        return rccs, rccs[:, :rank_cutoff].sum(axis=1) / maxauc
    # The rcc2d numba kernel already skips ranks >= rank_threshold, so the
    # padding (adding a column with total_genes and a 0.0 weight) is unnecessary.
    # This avoids two array copies (np.insert + np.append) per call.

    # Calculate recovery curves and AUCs.
    # For reason of generating the same results as in R we introduce an error by adding
    # one to the rank_cutoff for calculating the maximum AUC.
    if no_auc:
        rccs = _rcc2d_numba(rankings, weights, rank_threshold)
        return rccs, np.array([])
    maxauc = float((rank_cutoff + 1) * weights.sum())
    assert maxauc > 0
    # Fused kernel: computes RCCs and AUCs in a single pass
    rccs, aucs = _rcc2d_with_auc_numba(rankings, weights, rank_threshold, rank_cutoff, maxauc)
    return rccs, aucs


def enrichment4cells(
    rnk_mtx: pd.DataFrame, regulon: GeneSignature, auc_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Calculate the enrichment of the regulon for the cells in the ranking dataframe.

    :param rnk_mtx: The ranked expression matrix (n_cells, n_genes).
    :param regulon: The regulon the assess for enrichment
    :param auc_threshold: The fraction of the ranked genome to take into account for the
        calculation of the Area Under the recovery Curve.
    :return:
    """
    total_genes = len(rnk_mtx.columns)
    index = pd.MultiIndex.from_tuples(
        list(zip(rnk_mtx.index.values, repeat(regulon.name))), names=["Cell", "Regulon"]
    )
    rnk = rnk_mtx.iloc[:, rnk_mtx.columns.isin(regulon.genes)]
    if rnk.empty or (float(len(rnk.columns)) / float(len(regulon))) < 0.80:
        LOGGER.warning(
            f"Less than 80% of the genes in {regulon.name} are present in the "
            "expression matrix."
        )
        return pd.DataFrame(
            index=index,
            data={"AUC": np.zeros(shape=(rnk_mtx.shape[0]), dtype=np.float64)},
        )
    else:
        weights = np.asarray(
            [
                regulon[gene] if gene in regulon.genes else 1.0
                for gene in rnk.columns.values
            ]
        )
        return pd.DataFrame(
            index=index, data={"AUC": aucs(rnk, total_genes, weights, auc_threshold)}
        )


@jit(nopython=True, parallel=True, cache=True)
def _enrichment_column_layout(values: np.ndarray, aucs: np.ndarray, ness: np.ndarray) -> np.ndarray:
    """Copy private recovery values in tiles for column-oriented tabular output."""
    n_rows, n_cols = values.shape
    out = np.empty((n_cols + 2, n_rows), dtype=values.dtype).T
    for row in prange(n_rows):
        out[row, 0] = aucs[row]
        out[row, 1] = ness[row]
    tile = 32
    for block_col in prange((n_cols + tile - 1) // tile):
        col_start = block_col * tile
        for row_start in range(0, n_rows, tile):
            for col in range(col_start, min(col_start + tile, n_cols)):
                for row in range(row_start, min(row_start + tile, n_rows)):
                    out[row, col + 2] = values[row, col]
    return out


@jit(nopython=True, parallel=True, cache=True)
def _enrichment_rank_axis(rankings, weights, rank_threshold, rank_cutoff, maxauc):
    n_features, n_genes = rankings.shape
    values = np.empty((rank_threshold + 2, n_features), dtype=np.float64).T
    aucs = np.empty(n_features, dtype=np.float64)
    tile = 32
    for block in prange((n_features + tile - 1) // tile):
        start = block * tile
        rows = min(tile, n_features - start)
        counts = np.zeros((rank_threshold, rows), dtype=np.float64)
        for local in range(rows):
            for gene in range(n_genes):
                rank = rankings[start + local, gene]
                if rank < rank_threshold:
                    counts[rank, local] += weights[gene]
        cumulative = np.zeros(rows, dtype=np.float64)
        auc_sum = np.zeros(rows, dtype=np.float64)
        for rank in range(rank_threshold):
            for local in range(rows):
                cumulative[local] += counts[rank, local]
                values[start + local, rank + 2] = cumulative[local]
                if rank < rank_cutoff:
                    auc_sum[local] += cumulative[local]
        for local in range(rows):
            aucs[start + local] = auc_sum[local] / maxauc
            values[start + local, 0] = aucs[start + local]
    return values, aucs


def enrichment4features(
    rnkdb: RankingDatabase,
    gs: GeneSignature,
    rank_threshold: int = 5000,
    auc_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Calculate AUC and NES for all regulatory features in the supplied database using
    the genes of the given signature.

    :param rnkdb: The database.
    :param gs: The gene signature to assess for enrichment.
    :param rank_threshold: The total number of ranked genes to take into account when
        creating a recovery curve.
    :param auc_threshold: The fraction of the ranked genome to take into account for
        the calculation of the Area Under the recovery Curve.
    :return: A dataframe containing all information.
    """  # noqa: D205
    assert rnkdb, "A database must be supplied"
    assert gs, "A gene signature must be supplied"

    # Load rank of genes from database.
    df = rnkdb.load(gs)
    features = df.index.values
    genes = df.columns.values
    rankings = df.values
    gene2weight = gs.gene2weight
    weights = np.asarray([gene2weight[gene] for gene in genes])

    rank_cutoff = derive_rank_cutoff(auc_threshold, rnkdb.total_genes, rank_threshold)
    if _kernel_inputs_supported(rankings, weights):
        maxauc = float((rank_cutoff + 1) * weights.sum())
        assert maxauc > 0
        enrichment_values, aucs = _enrichment_rank_axis(
            rankings, weights, rank_threshold, rank_cutoff, maxauc
        )
    else:
        rccs, aucs = recovery(df, rnkdb.total_genes, weights, rank_threshold, auc_threshold)
        enrichment_values = None
    # NES uses the population standard deviation (ddof=0), not the sample
    # standard deviation. Preserve NumPy's reduction/degenerate-case behavior.
    ness = (aucs - aucs.mean()) / aucs.std()

    if enrichment_values is None:
        enrichment_values = _enrichment_column_layout(rccs, aucs, ness)
    else:
        enrichment_values[:, 1] = ness

    # The creation of a dataframe is a severe performance penalty.
    # Use copy=False for large arrays to avoid unnecessary copies.
    df_enrichment = pd.DataFrame(
        index=features,
        columns=pd.MultiIndex.from_tuples(
            [("Enrichment", "AUC"), ("Enrichment", "NES")] +
            [("Recovery", r) for r in range(rank_threshold)]
        ),
        data=enrichment_values,
        copy=False,
    )
    df_rnks = pd.DataFrame(
        index=features,
        columns=pd.MultiIndex.from_product([["Ranking"], genes]),
        data=rankings.copy(),
        copy=False,
    )
    return pd.concat([df_enrichment, df_rnks], axis=1, copy=False)


def leading_edge(
    rcc: np.ndarray,
    avg2stdrcc: np.ndarray,
    ranking: np.ndarray,
    genes: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[list[tuple[str, float]], int]:
    """
    Calculate the leading edge for a given recovery curve.

    :param rcc: The recovery curve.
    :param avg2stdrcc: The average + 2 standard deviation recovery curve.
    :param ranking: The rank numbers of the gene signature for a given regulatory
        feature.
    :param genes: The genes corresponding to the ranking available in the
         aforementioned parameter.
    :param weights: The weights for these genes.
    :return: The leading edge returned as a list of tuple. Each tuple associates a
        gene part of the leading edge with its rank or with its importance (if gene
        signature supplied). In addition, the rank at maximum difference is returned.
    """

    def critical_point() -> tuple[int, int]:
        """Returns (rank_at_max, max_recovery)."""
        rank_at_max = np.argmax(rcc - avg2stdrcc)
        return rank_at_max, rcc[rank_at_max]

    def get_genes(rank_at_max: int) -> list[tuple[str, float]]:
        sorted_idx = np.argsort(ranking)
        sranking = ranking[sorted_idx]
        gene_ids = genes[sorted_idx]
        # Make sure to include the gene at the leading edge itself. This is different
        # from the i-cisTarget implementation but is inline with the RcisTarget
        # implementation.
        filtered_idx = sranking <= rank_at_max
        filtered_gene_ids = gene_ids[filtered_idx]
        return list(
            zip(
                filtered_gene_ids,
                weights[sorted_idx][filtered_idx]
                if weights is not None
                else sranking[filtered_idx],
            )
        )

    rank_at_max, _n_recovered_genes = critical_point()
    # noinspection PyTypeChecker
    return get_genes(rank_at_max), rank_at_max


def leading_edge4row(
    row: pd.Series,
    avg2stdrcc: np.ndarray,
    genes: np.ndarray,
    weights: np.ndarray | None = None,
) -> pd.Series:
    """
    Calculate the leading edge for a row of a dataframe.

    Should be used with partial function application to make this
    function amenable to the apply idiom common for dataframes.

    :param row: The row of the dataframe to calculate the leading edge for.
    :param avg2stdrcc: The average + 2 standard deviation recovery curve.
    :param genes: The genes corresponding to the ranking available in the supplied row.
    :param weights: The weights for these genes.
    :return: The leading edge returned as a list of tuple. Each tuple associates a gene
        part of the leading edge with its rank or with its importance (if gene signature
        supplied).
    """
    return pd.Series(
        data=leading_edge(
            row["Recovery"].values, avg2stdrcc, row["Ranking"].values, genes, weights
        )
    )


# Giving numba a signature makes the code marginally faster but with losing flexibility
# (only being able to use one type of integers used in rankings).
# @jit(signature_or_function=float64(int16[:], int_, float64), nopython=True)
@jit(nopython=True, cache=True)
def auc1d(ranking: np.ndarray, rank_cutoff: int, max_auc: float) -> float:
    """
    Calculate the AUC of the recovery curve of a single ranking. [DEPRECATED].

    :param ranking: The rank numbers of the genes.
    :param rank_cutoff: The maximum rank to take into account when calculating the AUC.
    :param max_auc: The maximum AUC.
    :return: The normalized AUC.
    """
    # Using concatenate and full constructs required by numba.
    # The rankings are 0-based. The position at the rank threshold is included in the
    # calculation.
    x = np.concatenate(
        (
            np.sort(ranking[ranking < rank_cutoff]),
            np.full((1,), rank_cutoff, dtype=np.int_),
        )
    )
    y = np.arange(1, x.size, dtype=np.float64)
    return np.sum(np.diff(x) * y) / max_auc


@jit(nopython=True, cache=True)
def weighted_auc1d(
    ranking: np.ndarray, weights: np.ndarray, rank_cutoff: int, max_auc: float
) -> np.ndarray:
    """
    Calculate the AUC of the weighted recovery curve of a single ranking.

    :param ranking: The rank numbers of the genes.
    :param weights: The associated weights.
    :param rank_cutoff: The maximum rank to take into account when calculating the AUC.
    :param max_auc: The maximum AUC.
    :return: The normalized AUC.
    """
    # Using concatenate and full constructs required by numba.
    # The rankings are 0-based. The position at the rank threshold is included in the
    # calculation.
    filter_idx = ranking < rank_cutoff
    x = ranking[filter_idx]
    y = weights[filter_idx]
    sort_idx = np.argsort(x)
    x = np.concatenate((x[sort_idx], np.full((1,), rank_cutoff, dtype=np.int_)))
    y = y[sort_idx].cumsum()
    return np.sum(np.diff(x) * y) / max_auc


@jit(nopython=True, cache=True)
def _auc2d_numba_serial(
    rankings: np.ndarray, weights: np.ndarray, rank_cutoff: int, max_auc: float
) -> np.ndarray:
    """
    Serial numba 2D AUC kernel for small n_features (avoids parallel overhead).
    """
    n_features = rankings.shape[0]
    n_genes = rankings.shape[1]
    aucs = np.empty(shape=(n_features,), dtype=np.float64)
    max_filtered = n_genes + 1
    x_tmp = np.empty(max_filtered, dtype=np.int64)
    y_tmp = np.empty(max_filtered, dtype=np.float64)

    for row_idx in range(n_features):
        n_filtered = 0
        for j in range(n_genes):
            if rankings[row_idx, j] < rank_cutoff:
                x_tmp[n_filtered] = rankings[row_idx, j]
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


@jit(nopython=True, parallel=True, cache=True)
def _auc2d_numba(
    rankings: np.ndarray, weights: np.ndarray, rank_cutoff: int, max_auc: float
) -> np.ndarray:
    """
    Numba-compiled 2D AUC kernel. Replaces the Python loop calling
    weighted_auc1d per row with a single compiled pass.
    Uses parallel=True for multi-core execution.
    """
    n_features = rankings.shape[0]
    n_genes = rankings.shape[1]
    aucs = np.empty(shape=(n_features,), dtype=np.float64)

    for row_idx in prange(n_features):
        # Temp arrays for filtering and sorting (per-row in parallel mode)
        max_filtered = n_genes + 1
        x_tmp = np.empty(max_filtered, dtype=np.int64)
        y_tmp = np.empty(max_filtered, dtype=np.float64)
        # Filter genes below rank_cutoff
        n_filtered = 0
        for j in range(n_genes):
            if rankings[row_idx, j] < rank_cutoff:
                x_tmp[n_filtered] = rankings[row_idx, j]
                y_tmp[n_filtered] = weights[j]
                n_filtered += 1

        if n_filtered == 0:
            aucs[row_idx] = 0.0
            continue

        # Sort by rank value
        sort_idx = np.argsort(x_tmp[:n_filtered])

        # Compute AUC via cumulative sum of weights at sorted ranks
        total = 0.0
        prev_x = 0
        auc_sum = 0.0
        for i in range(n_filtered):
            curr_x = x_tmp[sort_idx[i]]
            if i > 0:
                auc_sum += (curr_x - prev_x) * total
            total += y_tmp[sort_idx[i]]
            prev_x = curr_x
        # Final segment to rank_cutoff
        auc_sum += (rank_cutoff - prev_x) * total

        aucs[row_idx] = auc_sum / max_auc

    return aucs


def auc2d(
    rankings: np.ndarray, weights: np.ndarray, rank_cutoff: int, max_auc: float
) -> np.ndarray:
    """
    Calculate the AUCs of multiple rankings.

    :param rankings: The rankings.
    :param weights: The weights associated with the selected genes.
    :param rank_cutoff: The maximum rank to take into account when calculating the AUC.
    :param max_auc: The maximum AUC.
    :return: The normalized AUCs.
    """
    # Use serial kernel for small n_features to avoid parallel overhead
    if rankings.shape[0] < 10000:
        return _auc2d_numba_serial(rankings, weights, rank_cutoff, max_auc)
    return _auc2d_numba(rankings, weights, rank_cutoff, max_auc)


def aucs(
    rnk: pd.DataFrame, total_genes: int, weights: np.ndarray, auc_threshold: float
) -> np.ndarray:
    """
    Calculate AUCs (implementation without calculating recovery curves first).

    :param rnk: A dataframe containing the rank number of genes of interest. Columns
        correspond to genes.
    :param total_genes: The total number of genes ranked.
    :param weights: The weights associated with the selected genes.
    :param auc_threshold: The fraction of the ranked genome to take into account for the
        calculation of the Area Under the recovery Curve.
    :return: An array with the AUCs.
    """
    rank_cutoff = derive_rank_cutoff(auc_threshold, total_genes)
    _features, _genes, rankings = rnk.index.values, rnk.columns.values, rnk.values
    y_max = weights.sum()
    # The rankings are 0-based. The position at the rank threshold is included in the
    # calculation.
    # The maximum AUC takes this into account.
    # For reason of generating the same results as in R we introduce an error by adding
    # one to the rank_cutoff for calculating the maximum AUC.
    maxauc = float((rank_cutoff + 1) * y_max)
    assert maxauc > 0
    return auc2d(rankings, weights, rank_cutoff, maxauc)
