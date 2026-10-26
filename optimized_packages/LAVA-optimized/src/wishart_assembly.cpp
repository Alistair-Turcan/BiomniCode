#include <cpp11.hpp>
#include <R_ext/BLAS.h>
#include <vector>
#include <memory>

using namespace cpp11;

// Bitwise-exact native replication of the per-draw assembly loop in
// matrixsampling::rwishart's noncentral branch (as used by LAVA's .rwishart_nc):
//   W[,,i] = (Theta + Sigma %*% Z[,,i]) %*% (Theta + t(Z[,,i]) %*% Sigma)
//            + tcrossprod(Sigma %*% WrootI[,,i])
// Every matrix product maps to the same BLAS dgemm call (same trans flags, same
// operand order, same leading dimensions) that R's %*% / crossprod / tcrossprod
// issue, and the elementwise adds are per-element IEEE-exact, so the output is
// bitwise identical to the R loop. No RNG is consumed here.
[[cpp11::register]]
cpp11::doubles wishart_assembly_nc(cpp11::doubles theta, cpp11::doubles sigma,
                                   cpp11::doubles z, cpp11::doubles wrooti,
                                   int p, int n) {
  const size_t pp = (size_t)p * (size_t)p;
  const double *th = REAL(theta);
  const double *S  = REAL(sigma);
  const double *Z  = REAL(z);
  const double *Wr = REAL(wrooti);

  cpp11::writable::doubles out((R_xlen_t)(pp * (size_t)n));
  double *O = REAL(out);

  std::vector<double> T1(pp), A(pp), T2(pp), B(pp), C(pp), T3(pp), D(pp);
  const double one = 1.0, zero = 0.0;

  for (int i = 0; i < n; i++) {
    const double *Zi = Z + (size_t)i * pp;
    const double *Wi = Wr + (size_t)i * pp;
    double *Oi = O + (size_t)i * pp;

    // T1 = Sigma %*% Z_i                      (R: %*%  -> dgemm N,N)
    F77_CALL(dgemm)("N", "N", &p, &p, &p, &one, S, &p, Zi, &p, &zero, T1.data(), &p FCONE FCONE);
    // A = Theta + T1
    for (size_t j = 0; j < pp; j++) A[j] = th[j] + T1[j];
    // T2 = t(Z_i) %*% Sigma                   (R: t() then %*%; == dgemm T,N on Z_i)
    F77_CALL(dgemm)("T", "N", &p, &p, &p, &one, Zi, &p, S, &p, &zero, T2.data(), &p FCONE FCONE);
    // B = Theta + T2
    for (size_t j = 0; j < pp; j++) B[j] = th[j] + T2[j];
    // C = A %*% B                             (dgemm N,N)
    F77_CALL(dgemm)("N", "N", &p, &p, &p, &one, A.data(), &p, B.data(), &p, &zero, C.data(), &p FCONE FCONE);
    // T3 = Sigma %*% WrootI_i                 (dgemm N,N)
    F77_CALL(dgemm)("N", "N", &p, &p, &p, &one, S, &p, Wi, &p, &zero, T3.data(), &p FCONE FCONE);
    // D = tcrossprod(T3)                      (R: tcrossprod -> dgemm N,T)
    F77_CALL(dgemm)("N", "T", &p, &p, &p, &one, T3.data(), &p, T3.data(), &p, &zero, D.data(), &p FCONE FCONE);
    // W_i = C + D
    for (size_t j = 0; j < pp; j++) Oi[j] = C[j] + D[j];
  }

  cpp11::writable::integers dim = {p, p, n};
  out.attr("dim") = dim;
  return out;
}

#include <Rmath.h>

// Probe: verify Rmath rchisq/rnorm called from C++ consume R's RNG stream
// identically to R's vectorized rchisq()/rnorm() calls.
[[cpp11::register]]
cpp11::writable::doubles rng_probe_cpp() {
  cpp11::writable::doubles out(3);
  out[0] = rchisq(5.0);
  out[1] = rnorm(0.0, 1.0);
  out[2] = rnorm(0.0, 1.0);
  return out;
}



// Shared RNG fill for the noncentral sampler: replicates the exact RNG consumption
// order of matrixsampling's rwishart_chol_I + Z fill:
//   for k in 1..p: sqrt(rchisq(n, nu-p-k+1)); then rnorm(p*(p-1)/2*n); then rnorm(p*p*n)
static void fill_wrooti_z(int p, int n, double nup, double* WrootI, double* Z) {
  // WrootI is fully written here (lower triangle explicitly zeroed), so the
  // caller may use uninitialized allocation; Z is fully overwritten as well.
  // RNG consumption order is unchanged: p x rchisq(n), rnorm(p(p-1)/2*n),
  // rnorm(p*p*n).
  const size_t pp = (size_t)p * (size_t)p;
  for (int k = 0; k < p; k++) {
    const double df = nup - k;
    for (int i = 0; i < n; i++)
      WrootI[(size_t)k + (size_t)k * p + (size_t)i * pp] = sqrt(rchisq(df));
  }
  for (size_t i = 0; i < (size_t)n; i++) {
    for (int col = 0; col < p; col++)
      for (int row = col + 1; row < p; row++)
        WrootI[(size_t)row + (size_t)col * p + i * pp] = rnorm(0.0, 1.0);
    for (int col = 0; col < p; col++)          // lower triangle: +0.0, same bits
      for (int row = col + 1; row < p; row++)  // as the previous zero-init
        WrootI[(size_t)col + (size_t)row * p + i * pp] = 0.0;
  }
  if (Z) for (size_t j = 0; j < pp * (size_t)n; j++) Z[j] = rnorm(0.0, 1.0);
}

// --- Scalar 3x3/2x2 matrix products replicating this machine's OpenBLAS dgemm ----
// Empirically established (bench/cpp_probe, 10M+ adversarial cases per combo):
// OpenBLAS 0.3.34 dgemm for these sizes computes, per output element, the
// zero-seeded forward FMA chain  c = fma(a0,b0,0); c = fma(a1,b1,c); ...
// This matches dgemm bitwise for all inputs except T,N results whose three
// products ALL underflow to zero (|a*b| < 2^-1074; sign-of-zero difference only)
// - unreachable with O(1) operands. Verified for p=2 and p=3, flag combos
// (N,N), (T,N), (N,T), including exact-zero column structures.
#pragma GCC push_options
#pragma GCC target("fma,avx2")
template<int P>
static inline void gemm_nn_scalar(const double* A, const double* B, double* C) {
  for (int j = 0; j < P; j++)
    for (int i = 0; i < P; i++) {
      double c = fma(A[i], B[j*P], 0.0);
      for (int k = 1; k < P; k++) c = fma(A[i + k*P], B[k + j*P], c);
      C[i + j*P] = c;
    }
}
template<int P>
static inline void gemm_tn_scalar(const double* A, const double* B, double* C) {
  for (int j = 0; j < P; j++)
    for (int i = 0; i < P; i++) {
      double c = fma(A[i*P], B[j*P], 0.0);
      for (int k = 1; k < P; k++) c = fma(A[k + i*P], B[k + j*P], c);
      C[i + j*P] = c;
    }
}
template<int P>
static inline void gemm_nt_scalar(const double* A, const double* B, double* C) {
  for (int j = 0; j < P; j++)
    for (int i = 0; i < P; i++) {
      double c = fma(A[i], B[j], 0.0);
      for (int k = 1; k < P; k++) c = fma(A[i + k*P], B[j + k*P], c);
      C[i + j*P] = c;
    }
}

template<int P>
static void assembly_loop_scalar(const double* th, const double* S, const double* Z,
                                 const double* Wr, double* O, int n) {
  constexpr int pp = P * P;
  for (int i = 0; i < n; i++) {
    const double* Zi = Z + (size_t)i * pp;
    const double* Wi = Wr + (size_t)i * pp;
    double* Oi = O + (size_t)i * pp;
    double T1[pp], A[pp], T2[pp], B[pp], C[pp], T3[pp], D[pp];
    gemm_nn_scalar<P>(S, Zi, T1);                       // Sigma %*% Z_i
    for (int j = 0; j < pp; j++) A[j] = th[j] + T1[j];
    gemm_tn_scalar<P>(Zi, S, T2);                       // t(Z_i) %*% Sigma
    for (int j = 0; j < pp; j++) B[j] = th[j] + T2[j];
    gemm_nn_scalar<P>(A, B, C);                         // A %*% B
    gemm_nn_scalar<P>(S, Wi, T3);                       // Sigma %*% WrootI_i
    gemm_nt_scalar<P>(T3, T3, D);                       // tcrossprod(T3)
    for (int j = 0; j < pp; j++) Oi[j] = C[j] + D[j];
  }
}

// per-draw W computation shared by the array-returning and fused paths
template<int P>
static inline void compute_w_scalar(const double* th, const double* S,
                                    const double* Zi, const double* Wi, double* Oi) {
  constexpr int pp = P * P;
  double T1[pp], A[pp], T2[pp], B[pp], C[pp], T3[pp], D[pp];
  gemm_nn_scalar<P>(S, Zi, T1);
  for (int j = 0; j < pp; j++) A[j] = th[j] + T1[j];
  gemm_tn_scalar<P>(Zi, S, T2);
  for (int j = 0; j < pp; j++) B[j] = th[j] + T2[j];
  gemm_nn_scalar<P>(A, B, C);
  gemm_nn_scalar<P>(S, Wi, T3);
  gemm_nt_scalar<P>(T3, T3, D);
  for (int j = 0; j < pp; j++) Oi[j] = C[j] + D[j];
}
#pragma GCC pop_options

// Full noncentral Wishart sampler (nu > 2p-1 branch of matrixsampling::rwishart)
// with the exact RNG consumption order of the R code:
//   for k in 1..p: sqrt(rchisq(n, nu-p-k+1)); rnorm(p*(p-1)/2*n); rnorm(p*p*n)
// followed by the bitwise-exact dgemm assembly (see wishart_assembly_nc).
[[cpp11::register]]
cpp11::doubles rwishart_nc_cpp(cpp11::doubles sigma_root, cpp11::doubles theta_root,
                               int n, double nu) {
  const int p = (int)std::sqrt((double)sigma_root.size());
  const size_t pp = (size_t)p * (size_t)p;
  const double *S = REAL(sigma_root);
  const double *th = REAL(theta_root);
  const double nup = nu - p;

  // --- rwishart_chol_I(n, nu - p, p): identical RNG stream, in-place fill ---
  std::vector<double> WrootI(pp * (size_t)n, 0.0);   // upper triangle stays 0
  for (int k = 0; k < p; k++) {                       // chisims[k,] = sqrt(rchisq(n, nup-k))
    const double df = nup - k;
    for (int i = 0; i < n; i++)
      WrootI[(size_t)k + (size_t)k * p + (size_t)i * pp] = sqrt(rchisq(df));
  }
  const size_t L = (size_t)p * (p - 1) / 2;
  for (size_t i = 0; i < (size_t)n; i++) {            // Y = rnorm(L*n), column-major per draw
    for (int col = 0; col < p; col++)
      for (int row = col + 1; row < p; row++)
        WrootI[(size_t)row + (size_t)col * p + i * pp] = rnorm(0.0, 1.0);
  }

  // --- Z = rnorm(p*p*n) ---
  std::vector<double> Z(pp * (size_t)n);
  for (size_t j = 0; j < pp * (size_t)n; j++) Z[j] = rnorm(0.0, 1.0);

  // --- per-draw assembly ---
  cpp11::writable::doubles out((R_xlen_t)(pp * (size_t)n));
  double *O = REAL(out);
  if (p == 3) {
    assembly_loop_scalar<3>(th, S, Z.data(), WrootI.data(), O, n);
  } else if (p == 2) {
    assembly_loop_scalar<2>(th, S, Z.data(), WrootI.data(), O, n);
  } else {
    // dgemm path (identical call sequence as wishart_assembly_nc) for other p
    std::vector<double> T1(pp), A(pp), T2(pp), B(pp), C(pp), T3(pp), D(pp);
    const double one = 1.0, zero = 0.0;
    int pi = p;
    for (int i = 0; i < n; i++) {
      const double *Zi = Z.data() + (size_t)i * pp;
      const double *Wi = WrootI.data() + (size_t)i * pp;
      double *Oi = O + (size_t)i * pp;
      F77_CALL(dgemm)("N", "N", &pi, &pi, &pi, &one, S, &pi, Zi, &pi, &zero, T1.data(), &pi FCONE FCONE);
      for (size_t j = 0; j < pp; j++) A[j] = th[j] + T1[j];
      F77_CALL(dgemm)("T", "N", &pi, &pi, &pi, &one, Zi, &pi, S, &pi, &zero, T2.data(), &pi FCONE FCONE);
      for (size_t j = 0; j < pp; j++) B[j] = th[j] + T2[j];
      F77_CALL(dgemm)("N", "N", &pi, &pi, &pi, &one, A.data(), &pi, B.data(), &pi, &zero, C.data(), &pi FCONE FCONE);
      F77_CALL(dgemm)("N", "N", &pi, &pi, &pi, &one, S, &pi, Wi, &pi, &zero, T3.data(), &pi FCONE FCONE);
      F77_CALL(dgemm)("N", "T", &pi, &pi, &pi, &one, T3.data(), &pi, T3.data(), &pi, &zero, D.data(), &pi FCONE FCONE);
      for (size_t j = 0; j < pp; j++) Oi[j] = C[j] + D[j];
    }
  }
  cpp11::writable::integers dim = {p, p, n};
  out.attr("dim") = dim;
  return out;
}


// Fused noncentral-Wishart sampler + bivariate conditional stats + conditional.norm
// for LAVA's bivariate.integral (p = 3, add.reverse = FALSE branch). Replicates,
// bitwise, the R pipeline:
//   draws = .rwishart_nc(n, K, sig.use, theta)
//   m = (draws[2,3,] + draws[1,3,] + sig.xys*(draws[1,2,] + draws[1,1,])) / K - sig.xy
//   v = var.y * (draws[2,2,] + 2*draws[1,2,] + draws[1,1,]) / K^2
//   v = ifelse(v <= 0, NA, sqrt(v))
//   conditional.norm(obs0, m, v)  = mean(pnorm(obs,m,v,F) + pnorm(-obs,m,v,T), na.rm=TRUE)
// W is never materialized; per-element FP operation order is identical to the R code.
// pnorm is R's own pnorm5 via Rmath; the mean uses a long double accumulator with
// extended-precision division, matching .Internal(mean). Emits "NaNs produced" once
// if any v < 0, matching R's vectorized sqrt warning.
[[cpp11::register]]
double fused_bivar_nc_cpp(cpp11::doubles sigma_root, cpp11::doubles theta_root,
                          int n, double nu, double K, double sig_xy, double sig_xys,
                          double var_y, double obs0) {
  const int p = 3;
  const size_t pp = 9;
  const double *S = REAL(sigma_root);
  const double *th = REAL(theta_root);
  const double nup = nu - p;

  // uninitialized allocation: fill_wrooti_z writes every element of both
  // buffers (WrootI lower triangle explicitly zeroed there)
  std::unique_ptr<double[]> WrootI(new double[pp * (size_t)n]);
  // iter22: Z is generated linearly (Z[j] = rnorm, j = 0..pp*n-1) and consumed
  // per draw (draw i reads Z[i*pp .. i*pp+pp-1], contiguous), so generation and
  // consumption interleave in fixed-size chunks with an identical RNG stream
  // and identical per-draw FP operations, while the live Z window shrinks from
  // pp*n doubles (65 MB at n=900k) to one chunk (8 MB).
  const size_t chunk_d = (size_t)1048576 / pp;          // doubles per chunk / pp
  std::unique_ptr<double[]> Z(new double[pp * chunk_d]);
  fill_wrooti_z(p, n, nup, WrootI.get(), nullptr);      // Z fill skipped (chunked below)

  const double K2 = K * K;
  const double obs = fabs(obs0);
  bool sqrt_warn = false;
  std::vector<double> pr((size_t)n);
  size_t z_filled = 0;                                   // draws generated so far
  for (int i = 0; i < n; i++) {
    size_t ci = (size_t)i % chunk_d;
    if (ci == 0) {                                       // (re)fill the chunk
      size_t todo = std::min(chunk_d, (size_t)n - z_filled);
      for (size_t j = 0; j < pp * todo; j++) Z[j] = rnorm(0.0, 1.0);
      z_filled += todo;
    }
    double W[9];
    compute_w_scalar<3>(th, S, Z.get() + ci * pp, WrootI.get() + (size_t)i * pp, W);
    // m (R eval order: (d23 + d13) + sig.xys*(d12 + d11), then /K, then -sig.xy)
    double m = ((W[7] + W[6]) + sig_xys * (W[3] + W[0])) / K - sig_xy;
    // v (R eval order: var.y * ((d22 + 2*d12) + d11), then /K^2)
    double v = var_y * ((W[4] + 2.0 * W[3]) + W[0]) / K2;
    double sd;
    if (v <= 0.0 || ISNAN(v)) { sd = NA_REAL; }
    else sd = sqrt(v);
    if (v < 0.0) sqrt_warn = true;   // R's vectorized sqrt(v) warns once if any v<0
    pr[(size_t)i] = pnorm(obs, m, sd, 0, 0) + pnorm(-obs, m, sd, 1, 0);
  }
  if (sqrt_warn) Rf_warning("NaNs produced");
  // mean(pr, na.rm=TRUE): R's real_mean (src/main/summary.c) is a TWO-PASS
  // compensated algorithm in 80-bit long double:
  //   pass 1: s = sum(x); s /= n
  //   pass 2: t = sum(x - s); s += t/n;  return (double)s
  // (na.rm filters NA/NaN first; both passes run over the filtered values in
  // order). The overflow fallback in real_mean is unreachable here because
  // 0 <= pr <= 2 and n < 2^31, so s is always finite.
  long double s = 0.0L;
  R_xlen_t c = 0;
  for (size_t i = 0; i < (size_t)n; i++) if (!ISNAN(pr[i])) { s += pr[i]; c++; }
  if (c == 0) return NA_REAL / NA_REAL;    // mean(numeric(0)) -> NaN
  s /= c;
  long double t = 0.0L;
  for (size_t i = 0; i < (size_t)n; i++) if (!ISNAN(pr[i])) t += (pr[i] - s);
  s += t / c;
  return (double)s;
}

