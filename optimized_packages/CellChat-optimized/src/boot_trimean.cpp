#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>

using namespace Rcpp;

// [biomni-code iter4] Grouped triMean kernel.
//
// Replicates, bitwise, the R-level computation
//     mean(collapse::fquantile(x, probs = c(0.25, 0.50, 0.50, 0.75), na.rm = TRUE))
// for each column of data_t restricted to the rows listed in each element of
// group_cells (1-based indices, as produced by split(seq_len(nC), group)).
//
// Bitwise-identity contract (verified against collapse 2.1.8 source
// src/fnth_fmedian_fquantile.c and base R 4.4 mean):
//  * type-7 quantile: h = (l - 1) * Q computed in double (exact for
//    Q in {0.25, 0.5, 0.75} and l < 2^53); ih = trunc(h); frac = h - ih;
//    a = ih-th order statistic (0-based); if ih == l-1 or frac <= 0 the
//    result is a, otherwise a + frac * (b - a) with b the (ih+1)-th order
//    statistic. Order statistics are algorithm-independent values, so
//    std::sort here is interchangeable with collapse's quickselect.
//  * na.rm = TRUE: NaN/NA values are dropped before ranking; if the
//    surviving count l <= 1 every quantile equals the surviving value
//    (or NA_REAL when l == 0), matching fquantile's early return.
//  * input length n <= 1: fquantile returns x[1] for all probs without
//    NA removal; replicated here.
//  * the final mean sums q25 + q50 + q50 + q75 sequentially in long double
//    (80-bit on x86-64) and divides by 4, matching R's .Internal(mean),
//    then rounds once to double.

static inline double trimean_col(std::vector<double>& buf, const double* col,
                                 const int* idx1, int n) {
  if (n <= 1) return n == 0 ? NA_REAL : col[idx1[0] - 1];
  buf.clear();
  for (int i = 0; i < n; ++i) {
    double v = col[idx1[i] - 1];
    if (!std::isnan(v)) buf.push_back(v);
  }
  const int l = (int)buf.size();
  if (l <= 1) return l == 0 ? NA_REAL : buf[0];
  double q[3];
  if (l < 8) {
    std::sort(buf.begin(), buf.end());
    static const double probs[3] = {0.25, 0.5, 0.75};
    for (int p = 0; p < 3; ++p) {
      const double h = (l - 1) * probs[p];
      const int ih = (int)h;
      const double frac = h - ih;
      const double a = buf[ih];
      if (ih == l - 1 || frac <= 0.0) {
        q[p] = a;
      } else {
        q[p] = a + frac * (buf[ih + 1] - a);
      }
    }
  } else {
    // [biomni-code iter15] adaptive quickselect quartiles. The type-7
    // quartile values are continuous functions of three order statistics
    // (and their upper neighbours when the fractional position is
    // non-zero); order statistics are algorithm-independent, so selecting
    // them with nth_element yields bitwise-identical q[] to the std::sort
    // path above (NaNs are removed at gather time; verified by the
    // correctness suite and benchmark digest). Median pivot first splits
    // the range so the Q1/Q3 selections run on halves: ~2l element ops
    // vs l*log2(l) for the full sort. Neighbour order statistics are
    // min scans over the provably-minimal ranges:
    //   b25 = min over (p25, p50]  (elements past p50 are >= v[p50])
    //   b50 = min over (p50, l)
    //   b75 = min over (p75, l)
    double* v = buf.data();
    const double h25 = (l - 1) * 0.25, h50 = (l - 1) * 0.5, h75 = (l - 1) * 0.75;
    const int p25 = (int)h25, p50 = (int)h50, p75 = (int)h75;
    const double f25 = h25 - p25, f50 = h50 - p50, f75 = h75 - p75;
    std::nth_element(v, v + p50, v + l);
    const double a50 = v[p50];
    std::nth_element(v, v + p25, v + p50);
    const double a25 = v[p25];
    std::nth_element(v + p50 + 1, v + p75, v + l);
    const double a75 = v[p75];
    q[0] = (f25 <= 0.0) ? a25
                        : a25 + f25 * (*std::min_element(v + p25 + 1, v + p50 + 1) - a25);
    q[1] = (f50 <= 0.0) ? a50
                        : a50 + f50 * (*std::min_element(v + p50 + 1, v + l) - a50);
    q[2] = (p75 == l - 1 || f75 <= 0.0) ? a75
                        : a75 + f75 * (*std::min_element(v + p75 + 1, v + l) - a75);
  }
  long double s = q[0];
  s += q[1];
  s += q[1];
  s += q[2];
  s /= 4.0L;
  return (double)s;
}

// data_t: cells x genes matrix (the transpose of the genes x cells data.use).
// group_cells: list of integer vectors of 1-based row indices into data_t.
// Returns a genes x G matrix: out[j, g] = triMean(data_t[group_cells[[g]], j]).
//[[Rcpp::export]]
NumericMatrix grouped_trimean_gt(NumericMatrix data_t, List group_cells) {
  const int G = group_cells.size();
  const int nGenes = data_t.ncol();
  const int nCells = data_t.nrow();
  NumericMatrix out(nGenes, G);
  std::vector<double> buf;
  buf.reserve(nCells > 0 ? nCells : 1);
  const double* base = REAL(data_t);
  double* obase = REAL(out);
  // [biomni-code iter16] j-outer/g-inner loop order: each column of
  // data_t stays cache-resident across all G group gathers instead of
  // being re-streamed from L3 once per group. Index pointers are hoisted
  // to avoid per-(j,g) Rcpp wrapper construction. Every output element is
  // computed by the same trimean_col from the same gathered values, so
  // results are bitwise-identical to the g-outer order.
  std::vector<const int*> ips(G);
  std::vector<int> ns(G);
  for (int g = 0; g < G; ++g) {
    IntegerVector cells = group_cells[g];
    ips[g] = INTEGER(cells);  // 1-based
    ns[g] = cells.size();
  }
  for (int j = 0; j < nGenes; ++j) {
    const double* col = base + (size_t)j * (size_t)nCells;
    for (int g = 0; g < G; ++g) {
      obase[(size_t)g * (size_t)nGenes + j] = trimean_col(buf, col, ips[g], ns[g]);
    }
  }
  return out;
}

// [biomni-code iter5] All-boots variant: one call returns the stacked
// genes x groups x nboot array (column-major: gene fastest, then group, then
// boot) -- identical layout to
// array(unlist(lapply(boot_cells, grouped_trimean_gt)), dim = c(nGenes, G, B)).
// boot_cells: list of nboot lists, each with G integer vectors of 1-based
// row indices into data_t.
//[[Rcpp::export]]
NumericVector boot_trimean_arr(NumericMatrix data_t, List boot_cells) {
  const int B = boot_cells.size();
  const int nGenes = data_t.ncol();
  const int nCells = data_t.nrow();
  const int G = Rf_length(boot_cells[0]);
  NumericVector out(Dimension(nGenes, G, B));
  std::vector<double> buf;
  buf.reserve(nCells > 0 ? nCells : 1);
  const double* base = REAL(data_t);
  double* obase = REAL(out);
  // [biomni-code iter16] j-outer/g-inner loop order (same rationale and
  // bitwise-identity argument as grouped_trimean_gt above): within a boot,
  // the G group index sets partition the cells, so streaming each column
  // once while gathering all G groups cuts L3 traffic ~G-fold versus
  // re-streaming the column per group.
  std::vector<const int*> ips(G);
  std::vector<int> ns(G);
  for (int b = 0; b < B; ++b) {
    List cells_g = boot_cells[b];
    for (int g = 0; g < G; ++g) {
      IntegerVector cells = cells_g[g];
      ips[g] = INTEGER(cells);  // 1-based
      ns[g] = cells.size();
    }
    for (int j = 0; j < nGenes; ++j) {
      const double* col = base + (size_t)j * (size_t)nCells;
      for (int g = 0; g < G; ++g) {
        obase[((size_t)b * G + g) * (size_t)nGenes + j] =
          trimean_col(buf, col, ips[g], ns[g]);
      }
    }
  }
  return out;
}
