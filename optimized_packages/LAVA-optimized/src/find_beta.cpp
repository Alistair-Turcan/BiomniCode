#include <cpp11.hpp>
#include <R_ext/BLAS.h>
#include <cmath>

using namespace cpp11;

// Bitwise-exact native replication of LAVA's find.beta() logistic-regression
// coefficient reconstruction (R/binary_processing.R), including the nested
// iterative searches find.b0 / make.b1 / update.b1 and the helpers
// stat.error / sum.error. Every floating-point operation mirrors the R code:
// libm exp/log/sqrt (same library R uses), double-precision sequential sums
// (matching R's sum() for length-3 vectors), and the same BLAS dgemm calls
// for t(x) %*% diag(w) %*% x. Control flow is replicated verbatim, so the
// returned c(b0, b1) is bitwise identical. No RNG is consumed.

// x = cbind(1, c(0,1,2)): 3x2 column-major; t(x): 2x3 column-major
static const double X_MAT[6]   = {1.0, 1.0, 1.0, 0.0, 1.0, 2.0};
static const double X_MAT_T[6] = {1.0, 0.0, 1.0, 1.0, 1.0, 2.0};
static const double X_COL2[3]  = {0.0, 1.0, 2.0};

static inline double r_sign(double x) {
  return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : x);  // NaN propagates like R's sign()
}

static inline double mu_k(double b0, double b1, double x) {
  return 1.0 / (1.0 + exp(-(b0 + b1 * x)));
}

// stat.error(b0, b1, stat, x, n.x)
static double stat_error(double b0, double b1, double stat, const double *nx) {
  double w[3], D[9] = {0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0};
  for (int k = 0; k < 3; k++) {
    double mu = mu_k(b0, b1, X_COL2[k]);
    w[k] = (mu * (1.0 - mu)) * nx[k];
  }
  D[0] = w[0]; D[4] = w[1]; D[8] = w[2];
  double C1[6], xsx[4];
  const double one = 1.0, zero = 0.0;
  int two = 2, three = 3;
  // t(x) %*% diag(w)
  F77_CALL(dgemm)("N", "N", &two, &three, &three, &one, X_MAT_T, &two, D, &three, &zero, C1, &two FCONE FCONE);
  // (...) %*% x
  F77_CALL(dgemm)("N", "N", &two, &two, &three, &one, C1, &two, X_MAT, &three, &zero, xsx, &two FCONE FCONE);
  double var = xsx[0] / (xsx[0] * xsx[3] - xsx[1] * xsx[1]);
  // ifelse(var > 0, abs(b1/sqrt(var)-stat)/abs(stat), Inf)
  return (var > 0.0) ? fabs(b1 / sqrt(var) - stat) / fabs(stat) : R_PosInf;
}

// sum.error(b0, b1, N1, n.x, val.x)
static double sum_error(double b0, double b1, double N1, const double *nx) {
  long double s = 0.0L;                    // R sum(): long double accumulator
  for (int k = 0; k < 3; k++) {
    double prod = mu_k(b0, b1, X_COL2[k]) * nx[k];   // double product, as in R
    s += prod;
  }
  double sd = (double)s;                   // sum() rounds to double
  return (sd - N1) / N1;
}

// find.b0(b0, b1, N1, n.x, val.x, step, reduction, tolerance)
static double find_b0(double b0, double b1, double N1, const double *nx,
                      double step, double reduction, double tolerance) {
  double err = sum_error(b0, b1, N1, nx);
  while (step != 0.0 && fabs(err) > tolerance) {
    double prop_b0 = b0 + (-step) * r_sign(err);
    double prop_err = sum_error(prop_b0, b1, N1, nx);
    if (fabs(prop_err) < fabs(err)) { b0 = prop_b0; err = prop_err; }
    else { step = step * reduction; }
  }
  return b0;
}

// make.b1: set b1 = value, refit b0 via find.b0, recompute err
static inline double make_b1(double cur_b0, double value, double stp,
                             double N1, const double *nx, double stat,
                             double reduction, double tolerance,
                             double &out_b0, double &out_err) {
  double nb1 = value;
  double nb0 = find_b0(cur_b0, nb1, N1, nx, fabs(stp), reduction,
                       fmax(fabs(stp), tolerance));
  out_b0 = nb0;
  out_err = stat_error(nb0, nb1, stat, nx);
  return nb1;
}

static void find_beta_core(const double* nx, double N1, double N_orig,
                           double stat, double tolerance, double reduction,
                           double* out) {
  double step = 0.01 * r_sign(stat);
  bool try_reverse = false;
  double b0 = log(N1 / (N_orig - N1));
  double b1 = 0.0, err = 0.0;
  b1 = make_b1(b0, 0.0, step, N1, nx, stat, reduction, tolerance, b0, err);
  while (fabs(step) > tolerance / 10.0 && err > tolerance) {
    double prop_b0 = 0.0, prop_err = 0.0;
    double prop_b1 = make_b1(b0, b1 + step, step, N1, nx, stat, reduction, tolerance,
                             prop_b0, prop_err);
    if (prop_err < err) {
      b0 = prop_b0; b1 = prop_b1; err = prop_err; try_reverse = false;
    } else {
      if (try_reverse) { step = -step; try_reverse = false; }
      else { step = step * reduction; try_reverse = (b1 != 0.0); }
    }
  }
  out[0] = b0; out[1] = b1;
}

[[cpp11::register]]
cpp11::writable::doubles find_beta_cpp(cpp11::doubles count_x, double N1, double N_orig,
                                       double stat, double tolerance, double reduction) {
  cpp11::writable::doubles out(2);
  find_beta_core(REAL(count_x), N1, N_orig, stat, tolerance, reduction, REAL(out));
  return out;
}

// Batch port of process.binary()'s per-SNP loop (R/binary_processing.R),
// replicating every floating-point operation and the exact result-length
// semantics of R's `corrs = c(); corrs[i] = v` idiom: the result has length
// max(assigned indices) with NA gaps, or NULL if no SNP was assigned; the
// final `corrs[!is.finite(corrs)] = NA` mask is applied to assigned entries.
// The caller keeps the vectorized preliminaries in R:
//   freq[freq > 0.5] = 1 - freq[freq > 0.5]; N.case = N * case.prop
[[cpp11::register]]
cpp11::sexp process_binary_cpp(cpp11::doubles stat, cpp11::doubles N,
                               cpp11::doubles freq, cpp11::doubles N_case) {
  const R_xlen_t n = stat.size();
  const double *st = REAL(stat), *Nv = REAL(N), *fr = REAL(freq), *Nc = REAL(N_case);
  cpp11::writable::doubles corrs(n);
  double* cp = REAL(corrs);
  for (R_xlen_t i = 0; i < n; i++) cp[i] = NA_REAL;
  R_xlen_t maxidx = 0;
  // iter25: per-SNP computations are independent (no RNG, no shared state;
  // find_beta_core touches only locals and read-only static const tables).
  // Each SNP's floating-point sequence is unchanged, so results are
  // bitwise-identical at any thread count. maxidx via max-reduction (integer
  // max is order-independent). Disjoint per-element writes through cp.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4) reduction(max:maxidx)
#endif
  for (R_xlen_t i = 0; i < n; i++) {
    const double f = fr[i];
    double cx[3];
    cx[0] = Nv[i] * ((1.0 - f) * (1.0 - f));   // N[i] * (1-freq[i])^2  (x^2 == x*x)
    cx[1] = Nv[i] * ((2.0 * f) * (1.0 - f));   // N[i] * 2*freq[i]*(1-freq[i]), left assoc
    cx[2] = Nv[i] * (f * f);                   // N[i] * freq[i]^2
    double beta[2];
    find_beta_core(cx, Nc[i], Nv[i], st[i], 1e-5, 0.25, beta);
    double mu[3];
    for (int k = 0; k < 3; k++) mu[k] = mu_k(beta[0], beta[1], X_COL2[k]);
    long double s = 0.0L;                      // R sum(): long double accumulator
    for (int k = 0; k < 3; k++) s += (X_COL2[k] * cx[k]) * mu[k];   // (x*count.x)*mu
    const double xty = (double)s;              // sum() rounds to double
    if (R_FINITE(xty)) {                       // is.finite(xty)
      long double s1 = 0.0L, s2 = 0.0L;
      for (int k = 0; k < 3; k++) s1 += cx[k] * X_COL2[k];              // sx
      for (int k = 0; k < 3; k++) s2 += cx[k] * (X_COL2[k] * X_COL2[k]); // sx2 (x^2 == x*x)
      const double sx = (double)s1, sx2 = (double)s2;
      const double sy = Nc[i];
      const double var_x = (sx2 - (sx * sx) / Nv[i]) / (Nv[i] - 1.0);
      const double var_y = (sy - (sy * sy) / Nv[i]) / (Nv[i] - 1.0);
      const double cov_xy = (xty - (sx * sy) / Nv[i]) / (Nv[i] - 1.0);
      if (var_x > 0.0 && var_y > 0.0) {
        cp[i] = cov_xy / sqrt(var_x * var_y);
        maxidx = i + 1;
      }
    }
  }
  if (maxidx == 0) {                           // corrs = c() -> NULL, then
    cpp11::writable::logicals empty((R_xlen_t)0);        // corrs[!is.finite(corrs)] = NA
    return cpp11::sexp(empty);                 // promotes NULL to logical(0)
  }
  for (R_xlen_t i = 0; i < maxidx; i++)        // corrs[!is.finite(corrs)] = NA
    if (!R_FINITE(cp[i])) cp[i] = NA_REAL;
  if (maxidx == n) return corrs;
  cpp11::writable::doubles out(maxidx);        // trailing unassigned SNPs: R's
  for (R_xlen_t i = 0; i < maxidx; i++) out[i] = corrs[i];  // vector is shorter
  return out;
}
