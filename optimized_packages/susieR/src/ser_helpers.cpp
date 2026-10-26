// SER hot-path helpers (iteration 8):
//   1. set_matrix_row_inplace  - write one row of a model matrix without the
//      copy-on-modify full-matrix duplicate. R's `model$alpha[l, ] <- v`
//      copies the whole L x p matrix because argument/list references keep
//      the matrix's REFCNT above 1 (REFCNT is conservative and stays
//      inflated after transient contact, so it cannot be used as a reliable
//      fast-path guard). The values written here are exactly the values the
//      R writeback would store, so results are bitwise-identical; the ONLY
//      semantic difference is aliasing. Callers are responsible for alias
//      safety: model$mu/mu2/lbf_variable are never aliased anywhere in the
//      package (verified by audit - all reads are row extractions or
//      transient full-matrix products), and model$alpha's only surviving
//      aliases (model$runtime$prev_alpha / alpha_history snapshots) are
//      tracked by an explicit data$fit_flags$alpha_shared flag that forces
//      the standard copy-on-modify writeback (which detaches the alias)
//      for the first writeback after each snapshot. The helper still
//      declines (returns false) on non-double, ALTREP, non-matrix, or
//      shape-mismatched targets, which fall back to the R writeback.
//   2. gaussian_ser_moments_cpp - fused elementwise posterior moments with
//      the identical per-element op order as the R reference
//      (post_var = V*shat2/(V+shat2); post_mean = post_var/shat2*betahat;
//      zeroing on non-finite inputs; post_mean2 = post_var + post_mean^2,
//      where R's `^2` is x*x). No reductions, so bitwise identity only
//      needs the same IEEE op order (this file is compiled with
//      -ffp-contract=off -fno-tree-vectorize -fno-tree-slp-vectorize).
//   3. colstats_scan_cpp - one-pass column means and centered sums of
//      squares for purity_scan_below, replicating base colMeans /
//      colSums(X^2) - n*cm^2 with long-double accumulators (base R's rsum
//      also accumulates in long double), plus an isnan scan equivalent to
//      anyNA(). The scan's only downstream use is a thresholded boolean
//      with a 1e-6 decision margin, so even a 1-ulp difference could not
//      change behavior; the accumulation order matches base R anyway.
#include <cpp11.hpp>
#include <cstring>
#include <cmath>

using namespace cpp11;

[[cpp11::register]]
bool set_matrix_row_inplace(cpp11::sexp model, std::string name, int l,
                            cpp11::sexp values) {
  if (TYPEOF(model) != VECSXP)
    return false;
  SEXP names = Rf_getAttrib(model, R_NamesSymbol);
  if (TYPEOF(names) != STRSXP)
    return false;
  R_xlen_t nf = XLENGTH(names);
  R_xlen_t idx = -1;
  for (R_xlen_t i = 0; i < nf; i++) {
    if (std::strcmp(CHAR(STRING_ELT(names, i)), name.c_str()) == 0) {
      idx = i;
      break;
    }
  }
  if (idx < 0)
    return false;
  SEXP mat = VECTOR_ELT(model, idx);
  // Non-double, ALTREP, or non-matrix targets keep the standard R
  // copy-on-modify writeback. Alias safety is the caller's responsibility
  // (see the file header): REFCNT cannot serve as a guard here because it
  // stays conservatively inflated after any transient contact.
  if (TYPEOF(mat) != REALSXP || ALTREP(mat))
    return false;
  SEXP dim = Rf_getAttrib(mat, R_DimSymbol);
  if (TYPEOF(dim) != INTSXP || XLENGTH(dim) != 2)
    return false;
  if (TYPEOF(values) != REALSXP || ALTREP(values))
    return false;
  R_xlen_t nr = INTEGER(dim)[0];
  R_xlen_t nc = INTEGER(dim)[1];
  if (l < 1 || (R_xlen_t)l > nr)
    return false;
  if (XLENGTH(values) != nc)
    return false;
  double* m = REAL(mat);
  const double* v = REAL(values);
  for (R_xlen_t j = 0; j < nc; j++)
    m[(l - 1) + j * nr] = v[j];
  return true;
}

[[cpp11::register]]
cpp11::list gaussian_ser_moments_cpp(cpp11::doubles betahat,
                                     cpp11::doubles shat2, double V) {
  R_xlen_t p = betahat.size();
  if (shat2.size() != p) {
    return cpp11::writable::list({"post_mean"_nm = R_NilValue,
                                  "post_mean2"_nm = R_NilValue,
                                  "ok"_nm = false});
  }
  cpp11::writable::doubles post_mean(p);
  cpp11::writable::doubles post_mean2(p);
  for (R_xlen_t j = 0; j < p; j++) {
    double b = betahat[j];
    double s = shat2[j];
    // Identical op order to the R reference:
    //   post_var  <- V * shat2 / (V + shat2)
    //   post_mean <- post_var / shat2 * betahat
    double pv = V * s / (V + s);
    double pm = pv / s * b;
    if (!std::isfinite(b) || !std::isfinite(s)) {
      pv = 0.0;
      pm = 0.0;
    }
    post_mean[j] = pm;
    post_mean2[j] = pv + pm * pm;  // R's `post_mean^2` is post_mean*post_mean
  }
  return cpp11::writable::list({"post_mean"_nm = post_mean,
                                "post_mean2"_nm = post_mean2,
                                "ok"_nm = true});
}

[[cpp11::register]]
cpp11::list colstats_scan_cpp(cpp11::doubles X, int n, int k) {
  // X holds the n x k column-major data of the subset matrix.
  if (n < 0 || k < 0 || X.size() != (R_xlen_t)n * (R_xlen_t)k) {
    return cpp11::writable::list({"cm"_nm = R_NilValue, "css"_nm = R_NilValue,
                                  "has_na"_nm = R_NilValue, "ok"_nm = false});
  }
  cpp11::writable::doubles cm(k);
  cpp11::writable::doubles css(k);
  bool has_na = false;
  const double* Xp = REAL(X);
  double nd = (double)n;
  for (R_xlen_t j = 0; j < k; j++) {
    const double* col = Xp + j * (R_xlen_t)n;
    long double s = 0.0L;
    long double s2 = 0.0L;
    for (R_xlen_t i = 0; i < n; i++) {
      double x = col[i];
      if (std::isnan(x))
        has_na = true;  // isnan covers both NA and NaN, matching anyNA()
      s += x;
      double sq = x * x;  // X^2 materializes doubles in R before colSums
      s2 += sq;
    }
    // base colMeans divides the long-double sum by n before casting to
    // double; matching that order keeps cm bitwise-identical to colMeans.
    double cmj = (double)(s / (long double)nd);
    cm[j] = cmj;
    // colSums(X^2) - n * cm^2, with R's `cm^2` = cm*cm evaluated first.
    css[j] = (double)s2 - nd * (cmj * cmj);
  }
  return cpp11::writable::list({"cm"_nm = cm, "css"_nm = css,
                                "has_na"_nm = has_na, "ok"_nm = true});
}

// Iteration 9 helpers (same FP discipline as above: elementwise op order
// matches the R reference exactly; reductions accumulate in long double in
// index order, matching base rsum; compiled with -ffp-contract=off
// -fno-tree-vectorize -fno-tree-slp-vectorize).
//
//   4. ser_stats_core_cpp - fused betahat/shat2/optim_init-max for
//      compute_ser_statistics.individual, replicating with identical
//      per-element op order:
//        betahat <- (1 / pw) * res     (R materializes 1/pw first)
//        shat2   <- sigma2 / pw, then shat2 * infl when infl is present
//        maxval  <- max(c(betahat^2 - shat2, 1), na.rm = TRUE)
//      R's `^2` is x*x; max(na.rm = TRUE) skips ISNAN values (NA and NaN);
//      the appended constant 1 is applied after the vector scan (max is
//      order-independent under exact IEEE comparisons). All ops are
//      elementwise or exact comparisons, so results are bitwise-identical
//      for every input; only shape/type mismatches decline.
//   5. ser_e_loglik_sums_cpp - the three reductions in the standard path of
//      SER_posterior_e_loglik.individual:
//        s1 = sum(rr * rr); s2 = sum(rr * xb);
//        s3 = sum(pw * (alpha[l,] * mu2[l,]))
//      Elementwise products are evaluated in double first (as R's `*` does),
//      then accumulated in long double in index order (as base rsum does),
//      so the returned doubles are bitwise-identical to the R chain. The
//      bracket arithmetic (s1 - 2*s2 + s3) and the scalar prefix stay in R.
//   6. xty_post_cpp - the post-matvec arithmetic of compute_Xty's dense
//      branch: as.numeric(t(ytX / csd) - cm / csd * sum(y)). ytX is the
//      1 x p crossprod output (column-major storage equals row order);
//      t() is an exact permutation and `cm / csd * sum(y)` parses as
//      (cm/csd) * sum(y), so the per-element op order below is
//      bitwise-identical.
//   7. colmeans_ranges_cpp - one fused sweep computing column means
//      (bitwise-identical to base colMeans: long-double accumulation,
//      division in long double before the double cast) and column ranges
//      (exact min/max; a column containing NA or NaN yields NA for both,
//      matching matrixStats::colRanges with na.rm = FALSE as consumed by
//      the constructor's NA screen). Replaces separate colRanges and
//      colMeans sweeps over X in individual_data_constructor.

[[cpp11::register]]
cpp11::list ser_stats_core_cpp(cpp11::doubles res, cpp11::doubles pw,
                               double sigma2, cpp11::doubles infl) {
  R_xlen_t p = res.size();
  if (pw.size() != p || (infl.size() != 1 && infl.size() != p))
    return cpp11::writable::list({"betahat"_nm = R_NilValue,
                                  "shat2"_nm = R_NilValue,
                                  "maxval"_nm = R_NilValue,
                                  "ok"_nm = false});
  cpp11::writable::doubles bh(p);
  cpp11::writable::doubles sh(p);
  const double* rp = REAL(res);
  const double* pp = REAL(pw);
  const double* ip = REAL(infl);
  bool infl_vec = (infl.size() == p);
  double m = R_NegInf;
  for (R_xlen_t j = 0; j < p; j++) {
    double inv = 1.0 / pp[j];   // R materializes (1 / pw) before multiplying
    double b = inv * rp[j];
    double s = sigma2 / pp[j];
    s = s * (infl_vec ? ip[j] : ip[0]);  // x * 1.0 == x exactly when no infl
    bh[j] = b;
    sh[j] = s;
    double t = b * b - s;        // R's betahat^2 is betahat*betahat
    if (!std::isnan(t) && t > m) // max(na.rm = TRUE) skips NA and NaN
      m = t;
  }
  if (1.0 > m)                   // the appended constant 1 of c(..., 1)
    m = 1.0;
  return cpp11::writable::list({"betahat"_nm = bh, "shat2"_nm = sh,
                                "maxval"_nm = m, "ok"_nm = true});
}

[[cpp11::register]]
cpp11::doubles ser_e_loglik_sums_cpp(cpp11::doubles rr, cpp11::doubles xb,
                                     cpp11::doubles pw, cpp11::doubles alpha,
                                     cpp11::doubles mu2, int l, int p) {
  R_xlen_t n = rr.size();
  R_xlen_t pd = (R_xlen_t)p;
  if (p < 1 || xb.size() != n || pw.size() != pd || l < 1 ||
      alpha.size() % pd != 0 || mu2.size() != alpha.size() ||
      (R_xlen_t)l > alpha.size() / pd)
    return cpp11::writable::doubles();  // length 0 -> R fallback
  const double* rp = REAL(rr);
  const double* xp = REAL(xb);
  const double* pp = REAL(pw);
  R_xlen_t L = alpha.size() / pd;
  const double* ap = REAL(alpha) + (l - 1);  // row l, column-major stride L
  const double* mp = REAL(mu2) + (l - 1);
  long double s1 = 0.0L, s2 = 0.0L, s3 = 0.0L;
  for (R_xlen_t i = 0; i < n; i++) {
    double t1 = rp[i] * rp[i];   // R materializes rr * rr as doubles first
    s1 += t1;
    double t2 = rp[i] * xp[i];
    s2 += t2;
  }
  for (R_xlen_t j = 0; j < pd; j++) {
    double eb2 = ap[j * L] * mp[j * L];  // alpha[l,] * mu2[l,] in double
    double t3 = pp[j] * eb2;             // pw * Eb2 in double
    s3 += t3;
  }
  return cpp11::writable::doubles({(double)s1, (double)s2, (double)s3});
}

[[cpp11::register]]
cpp11::doubles xty_post_cpp(cpp11::doubles ytX, cpp11::doubles cm,
                            cpp11::doubles csd, double sumy) {
  R_xlen_t p = csd.size();
  if (ytX.size() != p || cm.size() != p)
    return cpp11::writable::doubles();  // length 0 -> R fallback
  cpp11::writable::doubles out(p);
  const double* yp = REAL(ytX);
  const double* cp = REAL(cm);
  const double* sp = REAL(csd);
  for (R_xlen_t j = 0; j < p; j++) {
    double a = yp[j] / sp[j];    // ytX / csd (t() is an exact permutation)
    double t1 = cp[j] / sp[j];   // cm / csd
    double t2 = t1 * sumy;       // (cm / csd) * sum(y), R's parse order
    out[j] = a - t2;
  }
  return out;
}

[[cpp11::register]]
cpp11::list colmeans_ranges_cpp(cpp11::doubles X, int n, int p) {
  if (n < 1 || p < 1 || X.size() != (R_xlen_t)n * (R_xlen_t)p)
    return cpp11::writable::list({"cm"_nm = R_NilValue, "cmin"_nm = R_NilValue,
                                  "cmax"_nm = R_NilValue, "ok"_nm = false});
  cpp11::writable::doubles cm(p), cmin(p), cmax(p);
  const double* Xp = REAL(X);
  double nd = (double)n;
  for (R_xlen_t j = 0; j < p; j++) {
    const double* col = Xp + j * (R_xlen_t)n;
    long double s = 0.0L;
    double mn = R_PosInf, mx = R_NegInf;
    bool na = false;
    for (R_xlen_t i = 0; i < n; i++) {
      double x = col[i];
      s += x;
      if (std::isnan(x)) {  // isnan covers NA and NaN
        na = true;
      } else {
        if (x < mn) mn = x;
        if (x > mx) mx = x;
      }
    }
    cm[j] = (double)(s / (long double)nd);
    if (na) {
      cmin[j] = NA_REAL;
      cmax[j] = NA_REAL;
    } else {
      cmin[j] = mn;
      cmax[j] = mx;
    }
  }
  return cpp11::writable::list({"cm"_nm = cm, "cmin"_nm = cmin,
                                "cmax"_nm = cmax, "ok"_nm = true});
}

// out[j,l] = M[l,j] / csd[j] for a dense double L x p matrix M: bitwise
// t(M)/csd (t() is an exact permutation; the divide recycles csd down
// columns of t(M), i.e. element [j,l] is divided by csd[j]). Pure
// elementwise ops, no accumulation, so bitwise-identical for all inputs
// including NA/NaN/Inf. Returns an empty vector on shape mismatch so the
// caller can fall back to the R expression.
[[cpp11::register]]
cpp11::doubles mxt_pre_cpp(cpp11::doubles M, cpp11::doubles csd, int L, int p) {
  if (L < 1 || p < 1 || M.size() != (R_xlen_t)L * (R_xlen_t)p ||
      csd.size() != (R_xlen_t)p)
    return cpp11::writable::doubles();
  cpp11::writable::doubles out((R_xlen_t)p * (R_xlen_t)L);
  const double* Mp = REAL(M);
  const double* cp = REAL(csd);
  double* op = REAL(out);
  for (R_xlen_t l = 0; l < (R_xlen_t)L; l++)
    for (R_xlen_t j = 0; j < (R_xlen_t)p; j++)
      op[j + l * (R_xlen_t)p] = Mp[l + j * (R_xlen_t)L] / cp[j];
  return out;
}

// out[l,i] = tmp[i,l] - v[l] for a dense double n x L matrix tmp and
// L-vector v: bitwise t(tmp) - v (R recycles v down the columns of the
// L x n result, i.e. element [l,i] gets v[l]). Pure elementwise ops.
// Empty vector on shape mismatch -> caller falls back.
[[cpp11::register]]
cpp11::doubles mxt_post_cpp(cpp11::doubles tmp, cpp11::doubles v, int L, int n) {
  if (L < 1 || n < 1 || tmp.size() != (R_xlen_t)n * (R_xlen_t)L ||
      v.size() != (R_xlen_t)L)
    return cpp11::writable::doubles();
  cpp11::writable::doubles out((R_xlen_t)L * (R_xlen_t)n);
  const double* tp = REAL(tmp);
  const double* vp = REAL(v);
  double* op = REAL(out);
  for (R_xlen_t i = 0; i < (R_xlen_t)n; i++)
    for (R_xlen_t l = 0; l < (R_xlen_t)L; l++)
      op[l + i * (R_xlen_t)L] = tp[i + l * (R_xlen_t)n] - vp[l];
  return out;
}

// Bitwise scaled.Xb - sum(cm * b / csd) fused: R evaluates (cm * b) / csd
// elementwise in double precision, then sum() accumulates sequentially in
// long double (rsum), and the scalar result is subtracted from every
// element of scaled.Xb. This kernel keeps that exact per-element op order
// and accumulation precision. Empty vector on shape mismatch -> caller
// falls back to the R expression.
[[cpp11::register]]
cpp11::doubles xb_post_cpp(cpp11::doubles scaled_Xb, cpp11::doubles cm,
                           cpp11::doubles b, cpp11::doubles csd) {
  R_xlen_t n = scaled_Xb.size(), p = cm.size();
  if (p < 1 || b.size() != p || csd.size() != p)
    return cpp11::writable::doubles();
  const double* cmp = REAL(cm);
  const double* bp  = REAL(b);
  const double* cp  = REAL(csd);
  long double s = 0.0L;
  for (R_xlen_t j = 0; j < p; j++) {
    double t = (cmp[j] * bp[j]) / cp[j];
    s += t;
  }
  double sd = (double)s;
  cpp11::writable::doubles out(n);
  const double* xp = REAL(scaled_Xb);
  double* op = REAL(out);
  for (R_xlen_t i = 0; i < n; i++)
    op[i] = xp[i] - sd;
  return out;
}
