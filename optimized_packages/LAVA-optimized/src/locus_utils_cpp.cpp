#include <cpp11.hpp>
#include <R_ext/Arith.h>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <string>
#include <vector>

using namespace cpp11;

// ---------------------------------------------------------------------------
// Flat hash table over interned CHARSXP pointers (power-of-2 open addressing,
// multiplicative mix). CHARSXPs are interned in R's global string pool, so
// pointer equality is exactly content equality.
struct CharSet {
  std::vector<SEXP> keys;
  size_t mask;
  bool has_na = false;
  bool ok = true;   // false if table contains non-ASCII bytes

  static inline size_t mix(SEXP p) {
    uintptr_t v = (uintptr_t)p >> 4;
    return (size_t)((v * 0x9E3779B97F4A7C15ULL) >> 16);
  }
  explicit CharSet(R_xlen_t n) {
    size_t m = 16;
    while (m < (size_t)n * 2) m <<= 1;   // iter22: load factor <= 0.5 (membership result is logical - any load factor is bitwise-identical)
    keys.assign(m, nullptr);
    mask = m - 1;
  }
  inline void insert(SEXP s) {
    size_t h = mix(s) & mask;
    while (keys[h] && keys[h] != s) h = (h + 1) & mask;
    keys[h] = s;
  }
  inline bool contains(SEXP s) const {
    size_t h = mix(s) & mask;
    while (keys[h]) { if (keys[h] == s) return true; h = (h + 1) & mask; }
    return false;
  }
};

// Bitwise-identical replacement for base `%in%` (match(x, table, nomatch=0L) > 0L)
// on character vectors. The result is logical (no floating point), so any
// correct membership test is bitwise-identical to base. NA matches NA exactly
// as match() does. Returns NULL if the table has non-ASCII bytes or inputs are
// not character (R wrapper falls back to base %in%). Raw SEXP API: the
// cpp11::strings element-access layer measured ~8x slower (20ms vs 2.4ms per
// 84k x 2k membership; bench/cpp_probe/membership_probe.cpp).
[[cpp11::register]]
cpp11::sexp in_set_cpp(cpp11::sexp x_in, cpp11::sexp table_in) {
  if (TYPEOF(x_in) != STRSXP || TYPEOF(table_in) != STRSXP) return R_NilValue;
  CharSet cs(Rf_xlength(table_in));
  {
    const SEXP* tp = STRING_PTR(table_in);
    const R_xlen_t nt = Rf_xlength(table_in);
    for (R_xlen_t i = 0; i < nt; i++) {
      SEXP ts = tp[i];
      if (ts == NA_STRING) { cs.has_na = true; continue; }
      const unsigned char *c = (const unsigned char *)CHAR(ts);
      for (const unsigned char *q = c; *q; q++) if (*q >= 0x80) return R_NilValue;
      cs.insert(ts);
    }
  }
  const SEXP* xp = STRING_PTR(x_in);
  const R_xlen_t n = Rf_xlength(x_in);
  SEXP out = PROTECT(Rf_allocVector(LGLSXP, n));
  int* op = LOGICAL(out);
  for (R_xlen_t i = 0; i < n; i++) {
    SEXP xs = xp[i];
    op[i] = (xs == NA_STRING) ? (cs.has_na ? 1 : 0) : (cs.contains(xs) ? 1 : 0);
  }
  UNPROTECT(1);
  return cpp11::sexp(out);
}

// which(in_set_cpp(x, table)) fused: returns the 1-based indices of x that are
// members of table (NA in x counts only when table contains NA), identical to
// base which(x %in% table) - an integer index set, no floating point. NULL on
// non-ASCII table or non-character input, as in in_set_cpp (R wrapper falls
// back).
[[cpp11::register]]
cpp11::sexp which_in_set_cpp(cpp11::sexp x_in, cpp11::sexp table_in) {
  if (TYPEOF(x_in) != STRSXP || TYPEOF(table_in) != STRSXP) return R_NilValue;
  CharSet cs(Rf_xlength(table_in));
  {
    const SEXP* tp = STRING_PTR(table_in);
    const R_xlen_t nt = Rf_xlength(table_in);
    for (R_xlen_t i = 0; i < nt; i++) {
      SEXP ts = tp[i];
      if (ts == NA_STRING) { cs.has_na = true; continue; }
      const unsigned char *c = (const unsigned char *)CHAR(ts);
      for (const unsigned char *q = c; *q; q++) if (*q >= 0x80) return R_NilValue;
      cs.insert(ts);
    }
  }
  const SEXP* xp = STRING_PTR(x_in);
  const R_xlen_t n = Rf_xlength(x_in);
  std::vector<int> idx;
  for (R_xlen_t i = 0; i < n; i++) {
    SEXP xs = xp[i];
    if (xs == NA_STRING) { if (cs.has_na) idx.push_back((int)(i + 1)); }
    else if (cs.contains(xs)) idx.push_back((int)(i + 1));
  }
  SEXP out = PROTECT(Rf_allocVector(INTSXP, (R_xlen_t)idx.size()));
  int* op = INTEGER(out);
  for (size_t k = 0; k < idx.size(); k++) op[k] = idx[k];
  UNPROTECT(1);
  return cpp11::sexp(out);
}

// ---------------------------------------------------------------------------
// Blocked matrix transpose: a pure permutation of doubles (no floating-point
// operations), hence bitwise-identical to base t() for any input, including
// NA/NaN/Inf/-0.0. Used to replace t(La.svd(...)$vt) inside svd(X, nu=0).
[[cpp11::register]]
cpp11::writable::doubles_matrix<> transpose_cpp(cpp11::doubles_matrix<> m) {
  const int nr = m.nrow(), nc = m.ncol();
  cpp11::writable::doubles_matrix<> out(nc, nr);
  const double *ip = REAL(m);
  double *op = REAL(out);
  const int B = 64;   // iter22: tuned (pure permutation - any block size is bitwise-identical)
  // iter24: parallel over row blocks (no floating-point operations at all, so
  // any scheduling is bitwise-identical).
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
  for (int ii = 0; ii < nr; ii += B) {
    const int imax = std::min(ii + B, nr);
    for (int jj = 0; jj < nc; jj += B) {
      const int jmax = std::min(jj + B, nc);
      for (int i = ii; i < imax; i++)
        for (int j = jj; j < jmax; j++)
          op[j + (size_t)i * nc] = ip[i + (size_t)j * nr];  // out[j,i] = m[i,j]
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Fused equivalent of { X = scale_zero_na_cpp(x); X = scale_cpp(X) } as used in
// decompose.ld(): phase A replicates scale_zero_na_cpp exactly (scale.default
// semantics with long double accumulators, NaN/NA -> +0.0 in the final sweep);
// phase B replicates scale_cpp exactly on the phase-A result, in place.
// Per-element and per-column floating-point operation order is unchanged; one
// 16 MB intermediate allocation and one .Call round-trip are saved.
[[cpp11::register]]
cpp11::writable::doubles_matrix<> scale_zero_na_scale_cpp(cpp11::sexp x_in) {
  SEXP xs = Rf_coerceVector(x_in, REALSXP);
  cpp11::doubles_matrix<> x = cpp11::as_cpp<cpp11::doubles_matrix<>>(xs);
  const int nr = x.nrow(), nc = x.ncol();
  const double *xp = REAL(x);
  cpp11::writable::doubles_matrix<> out(nr, nc);
  double *op = REAL(out);

  // iter24: the six per-column passes (phase A mean/center+var/divide, then
  // phase B mean/center+var/divide on the phase-A result) are fused into one
  // per-column pipeline and parallelized across columns with OpenMP static
  // scheduling. Each column's floating-point operations and their order are
  // exactly the unfused sequential ones (per-column long-double accumulators,
  // same sweep order), so the result is bitwise-identical for any thread
  // count. No R API calls inside the parallel region.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
  for (int j = 0; j < nc; j++) {
    const double *col = xp + (size_t)j * nr;
    double *ocol = op + (size_t)j * nr;
    // phase A: scale_zero_na_cpp semantics
    long double s = 0.0L; int cnt = 0;
    for (int i = 0; i < nr; i++) {
      double v = col[i];
      if (ISNAN(v)) continue;
      s += v; cnt++;
    }
    const double cj = (double)(s / (long double)cnt);
    long double s2 = 0.0L; int cnt2 = 0;
    for (int i = 0; i < nr; i++) {
      double v = col[i] - cj;
      ocol[i] = v;
      if (ISNAN(v)) continue;
      double vv = v * v;
      s2 += vv; cnt2++;
    }
    int denom = cnt2 - 1; if (denom < 1) denom = 1;
    const double scj = sqrt((double)s2 / (double)denom);
    for (int i = 0; i < nr; i++) {
      double v = ocol[i] / scj;
      ocol[i] = ISNAN(v) ? 0.0 : v;
    }
    // phase B: scale_cpp semantics on the phase-A result
    long double t = 0.0L; int ccnt = 0;
    for (int i = 0; i < nr; i++) {
      double v = ocol[i];
      if (ISNAN(v)) continue;
      t += v; ccnt++;
    }
    const double dj = (double)(t / (long double)ccnt);
    long double t2 = 0.0L; int ccnt2 = 0;
    for (int i = 0; i < nr; i++) {
      double v = ocol[i] - dj;
      ocol[i] = v;
      if (ISNAN(v)) continue;
      double vv = v * v;
      t2 += vv; ccnt2++;
    }
    int denom2 = ccnt2 - 1; if (denom2 < 1) denom2 = 1;
    const double sdj = sqrt((double)t2 / (double)denom2);
    for (int i = 0; i < nr; i++) ocol[i] = ocol[i] / sdj;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Bitwise-identical port of stats::quantile(x, c(0.025, 0.975), na.rm=TRUE)
// type 7 for a numeric vector, replicating R 4.4's quantile.default exactly:
//   x <- x[!is.na(x)]; n <- length(x)
//   index <- 1 + max(n-1,0) * probs; lo <- floor(index); hi <- ceiling(index)
//   qs <- sort(x)[lo];  if (index > lo && x[hi] != qs) qs <- (1-h)*qs + h*x[hi]
// The partial sort is replaced by nth_element: order statistics are exact
// values regardless of selection algorithm. The interpolation expression and
// its guards are replicated verbatim (the x[hi] != qs guard matters: (1-h)*q +
// h*q is not guaranteed to round to q). Returns c(`2.5%`, `97.5%`) unnamed.
[[cpp11::register]]
cpp11::writable::doubles quantile7_pair_cpp(cpp11::sexp x_in) {
  SEXP xs = Rf_coerceVector(x_in, REALSXP);
  const R_xlen_t nx = Rf_xlength(xs);
  const double *xp = REAL(xs);
  std::vector<double> v;
  v.reserve(nx);
  for (R_xlen_t i = 0; i < nx; i++) {
    double val = xp[i];
    if (ISNAN(val)) continue;   // is.na drops NA and NaN
    v.push_back(val);
  }
  const double probs[2] = {0.025, 0.975};
  cpp11::writable::doubles out(2);
  const long n = (long)v.size();
  for (int k = 0; k < 2; k++) {
    if (n == 0) { out[k] = NA_REAL; continue; }   // x[lo] on empty -> NA
    double index = 1 + (double)std::max(n - 1, 0L) * probs[k];
    long lo = (long)floor(index), hi = (long)ceil(index);
    // 1-based lo/hi into sorted order; fetch both order statistics
    std::vector<double> w = v;   // work copy (nth_element reorders)
    double qlo, qhi;
    std::nth_element(w.begin(), w.begin() + (lo - 1), w.end());
    qlo = w[lo - 1];
    if (hi > lo) {
      std::nth_element(w.begin() + lo, w.begin() + (hi - 1), w.end());
      qhi = w[hi - 1];
    } else qhi = qlo;
    double qs = qlo;
    if (index > (double)lo && qhi != qs) {
      double h = index - (double)lo;
      qs = (1 - h) * qs + h * qhi;
    }
    out[k] = qs;
  }
  return out;
}

// ---------------------------------------------------------------------------
// iter16: ASCII case mapping, bitwise-identical to base tolower/toupper on
// pure-ASCII strings (for bytes < 0x80 R's case mapping is the ASCII
// A-Z<->a-z bijection in every locale). Returns NULL if any element contains
// a non-ASCII byte, where base may apply locale/Unicode mappings; the R
// wrapper falls back to base. NA -> NA. Created CHARSXPs are interned ASCII,
// so results are pointer-identical to base results.
static inline SEXP ascii_case_impl(SEXP x_in, bool upper) {
  if (TYPEOF(x_in) != STRSXP) return R_NilValue;
  const SEXP* xp = STRING_PTR(x_in);
  const R_xlen_t n = Rf_xlength(x_in);
  bool any_change = false;   // iter23: fused scan detects whether any byte would change
  for (R_xlen_t i = 0; i < n; i++) {
    SEXP s = xp[i];
    if (s == NA_STRING) continue;
    const unsigned char* q = (const unsigned char*)CHAR(s);
    const unsigned char* e = q + LENGTH(s);
    if (upper) { for (; q < e; q++) { unsigned char c = *q; if (c >= 0x80) return R_NilValue; any_change |= (c >= 'a' && c <= 'z'); } }
    else       { for (; q < e; q++) { unsigned char c = *q; if (c >= 0x80) return R_NilValue; any_change |= (c >= 'A' && c <= 'Z'); } }
  }
  // No byte needs changing: base tolower/toupper re-interns each unchanged
  // string, returning the identical CHARSXPs, so the input vector itself is
  // element-wise pointer-identical to the base result. Only returned when the
  // input is attribute-free, matching the attribute-less fresh vector below.
  if (!any_change && ATTRIB(x_in) == R_NilValue) return x_in;
  SEXP out = PROTECT(Rf_allocVector(STRSXP, n));
  std::string buf;
  for (R_xlen_t i = 0; i < n; i++) {
    SEXP s = xp[i];
    if (s == NA_STRING) { SET_STRING_ELT(out, i, NA_STRING); continue; }
    const char* c = CHAR(s);
    const R_xlen_t L = LENGTH(s);
    buf.assign(c, c + L);
    if (upper) { for (auto& ch : buf) if (ch >= 'a' && ch <= 'z') ch -= 32; }
    else       { for (auto& ch : buf) if (ch >= 'A' && ch <= 'Z') ch += 32; }
    SET_STRING_ELT(out, i, Rf_mkCharLen(buf.data(), (int)L));
  }
  UNPROTECT(1);
  return out;
}

[[cpp11::register]]
cpp11::sexp ascii_tolower_cpp(cpp11::sexp x_in) { return ascii_case_impl(x_in, false); }

[[cpp11::register]]
cpp11::sexp ascii_toupper_cpp(cpp11::sexp x_in) { return ascii_case_impl(x_in, true); }

// ---------------------------------------------------------------------------
// iter16: fused map.alleles (alignment.R). Replicates, bitwise:
//   a1 = match(toupper(a1), alleles); a2 = match(toupper(a2), alleles)
//   pair.index$code[a1 + (a2-1)*4]
// where alleles = c("A","C","G","T") and pair.index$code is the 16-vector of
// doubles below (strand-ambiguous pairs AT/TA/CG/GC set to NA). match()
// against 4 ASCII strings can never match a non-ASCII or multi-char string,
// and toupper() never maps a non-ASCII byte to A/C/G/T, so any element that
// is not exactly one ACTG letter (any case) maps to NA exactly as base does.
// NA -> NA. Output is REALSXP because pair.index$code is double.
[[cpp11::register]]
cpp11::sexp map_alleles_cpp(cpp11::sexp a1_in, cpp11::sexp a2_in) {
  if (TYPEOF(a1_in) != STRSXP || TYPEOF(a2_in) != STRSXP) return R_NilValue;
  const R_xlen_t n = Rf_xlength(a1_in);
  if (Rf_xlength(a2_in) != n) return R_NilValue;  // base would recycle; call sites never do
  // pair.index$code, 0-based index = (i1-1) + (i2-1)*4 with A=1,C=2,G=3,T=4
  static const double codes[16] = {
    NA_REAL, -1, -2, NA_REAL,   // a2=A: AA, CA, GA, TA(11->NA)
    1, NA_REAL, NA_REAL, 2,     // a2=C: AC, CC, GC(12->NA), TC
    2, NA_REAL, NA_REAL, 1,     // a2=G: AG, CG(12->NA), GG, TG
    NA_REAL, -2, -1, NA_REAL    // a2=T: AT(11->NA), CT, GT, TT
  };
  const SEXP* p1 = STRING_PTR(a1_in);
  const SEXP* p2 = STRING_PTR(a2_in);
  SEXP out = PROTECT(Rf_allocVector(REALSXP, n));
  double* o = REAL(out);
  for (R_xlen_t i = 0; i < n; i++) {
    SEXP s1 = p1[i], s2 = p2[i];
    int i1 = 0, i2 = 0;
    if (s1 != NA_STRING && LENGTH(s1) == 1) {
      unsigned char c = ((const unsigned char*)CHAR(s1))[0];
      if (c >= 'a' && c <= 'z') c -= 32;
      i1 = (c == 'A') ? 1 : (c == 'C') ? 2 : (c == 'G') ? 3 : (c == 'T') ? 4 : 0;
    }
    if (s2 != NA_STRING && LENGTH(s2) == 1) {
      unsigned char c = ((const unsigned char*)CHAR(s2))[0];
      if (c >= 'a' && c <= 'z') c -= 32;
      i2 = (c == 'A') ? 1 : (c == 'C') ? 2 : (c == 'G') ? 3 : (c == 'T') ? 4 : 0;
    }
    o[i] = (i1 && i2) ? codes[(i1 - 1) + (i2 - 1) * 4] : NA_REAL;
  }
  UNPROTECT(1);
  return out;
}

// ---------------------------------------------------------------------------
// iter17: fused colSums(x, na.rm=TRUE) + colSums(is.na(x)) in one pass.
// Bitwise-identical to R's do_colsum for a real matrix: per column, an
// 80-bit (LDOUBLE) accumulator over rows in sequential order, skipping
// NA/NaN (ISNAN covers both), result rounded to double on store - exactly
// R's algorithm. Counts are doubles, matching colSums() on a logical matrix. The caller keeps R's downstream arithmetic
// (sum / (nrow - na) / 2) in double, so the full expression is unchanged.
[[cpp11::register]]
cpp11::sexp colsums_na_cpp(cpp11::sexp x_in) {
  SEXP dim = Rf_getAttrib(x_in, R_DimSymbol);
  if (TYPEOF(dim) != INTSXP || Rf_xlength(dim) != 2) return R_NilValue;
  const R_xlen_t nr = INTEGER(dim)[0], nc = INTEGER(dim)[1];
  if (TYPEOF(x_in) == INTSXP) {
    // iter23: integer input (load_plink genotype matrix). Base colSums on
    // INTSXP accumulates in long double and returns double; colSums(is.na(x))
    // counts NA_INTEGER and returns double. Integer sums are exact in 80-bit,
    // so this is bitwise-identical to the base pair of calls.
    const int* x = INTEGER(x_in);
    SEXP sums = PROTECT(Rf_allocVector(REALSXP, nc));
    SEXP nas  = PROTECT(Rf_allocVector(REALSXP, nc));
    double* sp = REAL(sums); double* np = REAL(nas);
    // iter24: parallel across columns; per-column accumulation order unchanged
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
    for (R_xlen_t j = 0; j < nc; j++) {
      const int* col = x + j * nr;
      long double s = 0.0L;
      R_xlen_t na = 0;
      for (R_xlen_t i = 0; i < nr; i++) {
        int v = col[i];
        if (v == NA_INTEGER) na++; else s += (long double)v;
      }
      sp[j] = (double)s;
      np[j] = (double)na;
    }
    SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(out, 0, sums); SET_VECTOR_ELT(out, 1, nas);
    UNPROTECT(3);
    return out;
  }
  if (TYPEOF(x_in) != REALSXP) return R_NilValue;
  const double* x = REAL(x_in);
  SEXP sums = PROTECT(Rf_allocVector(REALSXP, nc));
  SEXP nas  = PROTECT(Rf_allocVector(REALSXP, nc));   // colSums(is.na(x)) returns double
  double* sp = REAL(sums); double* np = REAL(nas);
  // iter24: parallel across columns; per-column accumulation order unchanged
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(4)
#endif
  for (R_xlen_t j = 0; j < nc; j++) {
    const double* col = x + j * nr;
    long double s = 0.0L;
    R_xlen_t na = 0;
    for (R_xlen_t i = 0; i < nr; i++) {
      double v = col[i];
      if (ISNAN(v)) na++; else s += (long double)v;
    }
    sp[j] = (double)s;
    np[j] = (double)na;
  }
  SEXP out = PROTECT(Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(out, 0, sums); SET_VECTOR_ELT(out, 1, nas);
  UNPROTECT(3);
  return out;
}

