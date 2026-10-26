#include <cpp11.hpp>
#include <R_ext/Arith.h>
#include <cmath>
#include <vector>

using namespace cpp11;

// Bitwise-exact replication of base::scale.default(x, center=TRUE, scale=TRUE)
// for a numeric matrix, as used by LAVA's decompose.ld():
//   center_j = colMeans(x, na.rm=TRUE)   (double-sequential sum, skips NA and NaN,
//                                         divides by non-missing count; 0/0 = NaN)
//   x = sweep(x, 2, center, "-")         (per-element IEEE subtract)
//   scale_j = sqrt(sum(v^2)/max(1, length(v)-1))  over non-missing v of the
//                                             centered column (double-sequential)
//   x = sweep(x, 2, scale, "/")          (per-element IEEE divide)
// The "scaled:center"/"scaled:scale" attributes are not reproduced (unused by
// downstream code: only X[is.na(X)]=0 and svd(X) touch the result).
[[cpp11::register]]
cpp11::writable::doubles_matrix<> scale_cpp(cpp11::sexp x_in) {
  // scale.default starts with as.matrix(x) and its arithmetic coerces integer
  // input to double; coercing here is value-exact for integers (NA_integer_ ->
  // NA_real_) and a no-op for double input.
  SEXP xs = Rf_coerceVector(x_in, REALSXP);
  cpp11::doubles_matrix<> x = cpp11::as_cpp<cpp11::doubles_matrix<>>(xs);
  const int nr = x.nrow(), nc = x.ncol();
  const double *xp = REAL(x);
  cpp11::writable::doubles_matrix<> out(nr, nc);
  double *op = REAL(out);

  std::vector<double> center(nc);
  for (int j = 0; j < nc; j++) {
    const double *col = xp + (size_t)j * nr;
    long double s = 0.0L; int cnt = 0;   // R colMeans: long double accumulator
    for (int i = 0; i < nr; i++) {
      double v = col[i];
      if (ISNAN(v)) continue;          // colMeans(na.rm=TRUE) skips NA and NaN
      s += v; cnt++;
    }
    // R colMeans divides in extended precision, then rounds to double
    // (cnt==0 -> 0/0 = NaN, as in R)
    center[j] = (double)(s / (long double)cnt);
  }

  // subtract sweep fused with the v^2 accumulation: per-element operations and
  // per-column accumulation order are identical to the unfused sequence
  // (sweep "-", then sum(v^2) over non-missing v); one read pass over the
  // centered matrix is saved.
  std::vector<double> sc(nc);
  for (int j = 0; j < nc; j++) {
    const double *col = xp + (size_t)j * nr;
    double *ocol = op + (size_t)j * nr;
    const double cj = center[j];
    long double s = 0.0L; int cnt = 0;  // R sum(): long double accumulator
    for (int i = 0; i < nr; i++) {
      double v = col[i] - cj;          // per-element IEEE subtract, as in sweep
      ocol[i] = v;
      if (ISNAN(v)) continue;          // v[!is.na(v)] drops NA and NaN
      double vv = v * v;               // v^2 computed in double, as in R
      s += vv; cnt++;
    }
    int denom = cnt - 1; if (denom < 1) denom = 1;   // max(1, length(v)-1L)
    sc[j] = sqrt((double)s / (double)denom);         // sum() rounds to double first
  }
  for (int j = 0; j < nc; j++) {
    double *ocol = op + (size_t)j * nr;
    const double scj = sc[j];
    for (int i = 0; i < nr; i++) ocol[i] = ocol[i] / scj;
  }
  return out;
}

// scale.default(x, TRUE, TRUE) followed by x[is.na(x)] <- 0, fused: identical
// values to scale_cpp() with NaN/NA results replaced by +0.0 in the final
// sweep (is.na catches NA and NaN; ISNAN does too; -0.0 is preserved, as in R).
[[cpp11::register]]
cpp11::writable::doubles_matrix<> scale_zero_na_cpp(cpp11::sexp x_in) {
  SEXP xs = Rf_coerceVector(x_in, REALSXP);
  cpp11::doubles_matrix<> x = cpp11::as_cpp<cpp11::doubles_matrix<>>(xs);
  const int nr = x.nrow(), nc = x.ncol();
  const double *xp = REAL(x);
  cpp11::writable::doubles_matrix<> out(nr, nc);
  double *op = REAL(out);

  std::vector<double> center(nc);
  for (int j = 0; j < nc; j++) {
    const double *col = xp + (size_t)j * nr;
    long double s = 0.0L; int cnt = 0;
    for (int i = 0; i < nr; i++) {
      double v = col[i];
      if (ISNAN(v)) continue;
      s += v; cnt++;
    }
    center[j] = (double)(s / (long double)cnt);
  }
  std::vector<double> sc(nc);
  for (int j = 0; j < nc; j++) {
    const double *col = xp + (size_t)j * nr;
    double *ocol = op + (size_t)j * nr;
    const double cj = center[j];
    long double s = 0.0L; int cnt = 0;
    for (int i = 0; i < nr; i++) {
      double v = col[i] - cj;
      ocol[i] = v;
      if (ISNAN(v)) continue;
      double vv = v * v;
      s += vv; cnt++;
    }
    int denom = cnt - 1; if (denom < 1) denom = 1;
    sc[j] = sqrt((double)s / (double)denom);
  }
  for (int j = 0; j < nc; j++) {
    double *ocol = op + (size_t)j * nr;
    const double scj = sc[j];
    for (int i = 0; i < nr; i++) {
      double v = ocol[i] / scj;
      ocol[i] = ISNAN(v) ? 0.0 : v;    // x[is.na(x)] <- 0
    }
  }
  return out;
}
