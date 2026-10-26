// Fused SER kernel: gaussian_ser_lbf + lbf_stabilization + compute_posterior_weights
// in a single pass, bitwise-identical to the R reference chain:
//   shat2_safe <- pmax(shat2, .Machine$double.eps)
//   lbf <- -0.5 * log(1 + V / shat2_safe) + 0.5 * betahat^2 * V / (shat2_safe * (V + shat2_safe))
//   lbf[!is.finite(betahat) | !is.finite(shat2)] <- 0
//   lpo <- lbf + log_prior            (log_prior = log(pi + sqrt(eps)), precomputed in R)
//   w <- exp(lpo - max(lpo)); sw <- sum(w); alpha <- w / sw; lbf_model <- log(sw) + max(lpo)
//
// Bitwise identity with R requires (verified empirically on the build platform):
//   - identical IEEE op order per element, no FMA contraction
//     (this file is compiled with -ffp-contract=off via the per-file rule in Makevars)
//   - scalar libm log/exp, not auto-vectorized libmvec variants
//     (-fno-tree-vectorize -fno-tree-slp-vectorize via the same rule)
//   - b*b for betahat^2 (R's `^` special-cases exponent 2 to x*x; libm pow differs)
//   - sum accumulated in long double (R's rsum uses an LDOUBLE accumulator)
//   - any NaN in lpo or w, or any length mismatch -> ok = FALSE so the caller
//     falls back to the original R chain (R's max() NaN-payload semantics).
#include <cpp11.hpp>
#include <R_ext/RS.h>  // R_alloc
#include <cmath>
#include <limits>

using namespace cpp11;

[[cpp11::register]]
cpp11::list ser_lbf_kernel_cpp(cpp11::doubles betahat, cpp11::doubles shat2,
                               double V, cpp11::doubles log_prior,
                               bool lbf_model_only) {
  R_xlen_t p = betahat.size();
  const double eps = std::numeric_limits<double>::epsilon();

  // Length mismatch would make R recycle; decline to the R chain instead.
  if (shat2.size() != p || log_prior.size() != p) {
    return cpp11::writable::list({"lbf"_nm = R_NilValue, "alpha"_nm = R_NilValue,
                                  "lbf_model"_nm = R_NilValue, "ok"_nm = false});
  }

  // Lean path for prior-variance optimization: the caller only needs
  // lbf_model = log(sum(exp(lpo - max))) + max, so skip storing lbf and
  // alpha and skip the normalization division entirely. The max and the
  // long-double sum accumulate the same values in the same order as the
  // full path, so lbf_model is bitwise-identical either way.
  if (lbf_model_only) {
    // Scratch from R's alloc stack (freed at .Call return): avoids a
    // GC-tracked 8*p-byte allocation on every prior-variance optimization
    // evaluation. Same values, same order - bitwise-identical.
    double* lpo = (double*)R_alloc((size_t)p, sizeof(double));
    bool ok = true;
    double m = R_NegInf;
    for (R_xlen_t j = 0; j < p; j++) {
      double b = betahat[j];
      double s = shat2[j];
      double l;
      if (!std::isfinite(b) || !std::isfinite(s)) {
        l = 0.0;
      } else {
        double ssafe = s > eps ? s : eps;
        l = -0.5 * std::log(1.0 + V / ssafe) +
            0.5 * (b * b) * V / (ssafe * (V + ssafe));
      }
      double lp = l + log_prior[j];
      lpo[j] = lp;
      if (std::isnan(lp)) {
        ok = false;
      } else if (lp > m) {
        m = lp;
      }
    }
    if (!ok) {
      return cpp11::writable::list({"lbf"_nm = R_NilValue, "alpha"_nm = R_NilValue,
                                    "lbf_model"_nm = R_NilValue, "ok"_nm = false});
    }
    long double sum_w = 0.0L;
    for (R_xlen_t j = 0; j < p; j++) {
      double w = std::exp(lpo[j] - m);
      if (std::isnan(w)) {
        ok = false;
        break;
      }
      sum_w += w;
    }
    if (!ok) {
      return cpp11::writable::list({"lbf"_nm = R_NilValue, "alpha"_nm = R_NilValue,
                                    "lbf_model"_nm = R_NilValue, "ok"_nm = false});
    }
    double sw = (double)sum_w;
    double lbf_model = std::log(sw) + m;
    return cpp11::writable::list({"lbf"_nm = R_NilValue, "alpha"_nm = R_NilValue,
                                  "lbf_model"_nm = lbf_model, "ok"_nm = true});
  }

  cpp11::writable::doubles lbf(p);
  // Scratch lpo from R's alloc stack (see the lean path above).
  double* lpo = (double*)R_alloc((size_t)p, sizeof(double));

  bool ok = true;
  double m = R_NegInf;
  for (R_xlen_t j = 0; j < p; j++) {
    double b = betahat[j];
    double s = shat2[j];
    double l;
    if (!std::isfinite(b) || !std::isfinite(s)) {
      l = 0.0;
    } else {
      double ssafe = s > eps ? s : eps;  // pmax(s, eps) for finite s
      l = -0.5 * std::log(1.0 + V / ssafe) +
          0.5 * (b * b) * V / (ssafe * (V + ssafe));
    }
    lbf[j] = l;
    double lp = l + log_prior[j];
    lpo[j] = lp;
    if (std::isnan(lp)) {
      ok = false;  // R's max() NaN-payload semantics: defer to the R chain
    } else if (lp > m) {
      m = lp;
    }
  }

  if (!ok) {
    return cpp11::writable::list({"lbf"_nm = R_NilValue, "alpha"_nm = R_NilValue,
                                  "lbf_model"_nm = R_NilValue, "ok"_nm = false});
  }

  // w = exp(lpo - m); accumulate the sum in long double like R's rsum.
  cpp11::writable::doubles alpha(p);
  long double sum_w = 0.0L;
  for (R_xlen_t j = 0; j < p; j++) {
    double w = std::exp(lpo[j] - m);
    if (std::isnan(w)) {  // degenerate m = +/-Inf case
      ok = false;
      break;
    }
    alpha[j] = w;
    sum_w += w;
  }

  if (!ok) {
    return cpp11::writable::list({"lbf"_nm = R_NilValue, "alpha"_nm = R_NilValue,
                                  "lbf_model"_nm = R_NilValue, "ok"_nm = false});
  }

  double sw = (double)sum_w;
  for (R_xlen_t j = 0; j < p; j++) alpha[j] = alpha[j] / sw;
  double lbf_model = std::log(sw) + m;

  return cpp11::writable::list({"lbf"_nm = lbf, "alpha"_nm = alpha,
                                "lbf_model"_nm = lbf_model, "ok"_nm = true});
}
