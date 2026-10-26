#ifndef SFBM_H
#define SFBM_H

/******************************************************************************/

#ifndef STRICT_R_HEADERS
#define STRICT_R_HEADERS
#endif

#include <mio/mmap.hpp>
#include <system_error> // for std::error_code
#include <Rcpp.h>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define SFBM_HAVE_AVX2_PATH 1

// AVX2 compact AXPY: x[i0 + (k-lo)] += data[k] * coef over k in [lo, up).
// Separate _mm256_mul_pd/_mm256_add_pd (target "avx2" does NOT enable FMA,
// so no contraction is possible): each element gets exactly one IEEE mul and
// one IEEE add, bit-identical to the scalar loop. Prefetch geometry matches
// the scalar 8-wide path (512B/1KB/1.5KB ahead).
__attribute__((target("avx2")))
static void sfbm_axpy_avx2(const double* data, size_t lo, size_t up,
                           double* x, int i0, double coef) {
  size_t k = lo; int i = i0;
  const __m256d c = _mm256_set1_pd(coef);
  for (; k + 8 <= up; k += 8, i += 8) {
    __builtin_prefetch(&data[k + 64],  0, 1);
    __builtin_prefetch(&data[k + 128], 0, 1);
    __builtin_prefetch(&data[k + 192], 0, 1);
    __m256d d0 = _mm256_loadu_pd(&data[k]);
    __m256d d1 = _mm256_loadu_pd(&data[k + 4]);
    __m256d x0 = _mm256_loadu_pd(&x[i]);
    __m256d x1 = _mm256_loadu_pd(&x[i + 4]);
    x0 = _mm256_add_pd(x0, _mm256_mul_pd(d0, c));
    x1 = _mm256_add_pd(x1, _mm256_mul_pd(d1, c));
    _mm256_storeu_pd(&x[i], x0);
    _mm256_storeu_pd(&x[i + 4], x1);
  }
  for (; k < up; k++, i++) {
    __builtin_prefetch(&data[k + 64], 0, 1);
    x[i] += data[k] * coef;
  }
}

// AVX-512F compact AXPY: same structure, one 8-wide vector per iteration.
// Separate _mm512_mul_pd/_mm512_add_pd; per-element FP semantics identical
// to scalar. The empty asm barrier makes the mul result opaque so GCC cannot
// contract mul+add into vfmadd (bitwise identity checked by objdump and the
// correctness suite).
__attribute__((target("avx512f")))
static void sfbm_axpy_avx512(const double* data, size_t lo, size_t up,
                             double* x, int i0, double coef) {
  size_t k = lo; int i = i0;
  const __m512d c = _mm512_set1_pd(coef);
  for (; k + 8 <= up; k += 8, i += 8) {
    __builtin_prefetch(&data[k + 64],  0, 3);  // t0: fill L1 (stream is re-read every sweep)
    __builtin_prefetch(&data[k + 128], 0, 3);
    __builtin_prefetch(&data[k + 192], 0, 3);
    __m512d d  = _mm512_loadu_pd(&data[k]);
    __m512d xv = _mm512_loadu_pd(&x[i]);
    __m512d t  = _mm512_mul_pd(d, c);
    __asm__("" : "+x"(t));  // opaque identity: blocks FMA contraction
    _mm512_storeu_pd(&x[i], _mm512_add_pd(xv, t));
  }
  for (; k < up; k++, i++) {
    __builtin_prefetch(&data[k + 64], 0, 1);
    double t = data[k] * coef;
    __asm__("" : "+x"(t));  // opaque identity: blocks FMA contraction
    x[i] += t;
  }
}
#endif

/******************************************************************************/

class SFBM {
public:
  SFBM(std::string path, int n, int m, const std::vector<size_t>& p,
       const std::vector<int>& first_i)
  : n(n), m(m), p(p), first_i(first_i) {

    std::error_code error;
    this->ro_mmap.map(path, error);
    if (error)
      Rcpp::stop("Error when mapping file:\n  %s.\n", error.message());

    this->data = reinterpret_cast<const double*>(ro_mmap.data());

    this->compact = first_i.size() > 0;
  }

  const double* i_x() { return reinterpret_cast<const double*>(ro_mmap.data()); }
  size_t nrow() const { return n; }
  size_t ncol() const { return m; }
  bool is_compact() const { return compact; }

  template<class C>
  C prod(const C& x) {

    C res(n);
    for (int i = 0; i < n; i++) res[i] = 0;

    for (int j = 0; j < m; j++) {

      double x_j = x[j];

      if (x_j != 0) {

        if (compact) {

          size_t lo = p[j];
          size_t up = p[j + 1];
          int i = first_i[j];

          for (size_t k = lo; k < up; k++, i++)
            res[i] += data[k] * x_j;

        } else {

          size_t lo = 2 * p[j];
          size_t up = 2 * p[j + 1];

          for (size_t k = lo; k < up; k += 2) {
            int    ind = data[k];
            double val = data[k + 1];
            res[ind] += val * x_j;
          }

        }
      }
    }

    return res;
  }

  template<class C>
  C cprod(const C& x) {

    C res(m);

    for (int j = 0; j < m; j++)
      res[j] = dot_col(j, x);

    return res;
  }

  template<class C>
  double dot_col(int j, const C& x) {

    double cp = 0;

    if (compact) {

      size_t lo = p[j];
      size_t up = p[j + 1];

      size_t k = lo;
      int i = first_i[j];

      if (up >= (lo + 4)) {
        for (; k <= (up - 4); k += 4, i += 4) {  // unrolling optimization
          cp += (data[k] * x[i] + data[k + 1] * x[i + 1]) +
            (data[k + 2] * x[i + 2] + data[k + 3] * x[i + 3]);
        }
      }
      for (; k < up; k++, i++) cp += data[k] * x[i];

    } else {

      size_t lo = 2 * p[j];
      size_t up = 2 * p[j + 1];

      size_t k = lo;

      if (up >= (lo + 8)) {
        for (; k <= (up - 8); k += 8) {  // unrolling optimization
          cp += (data[k + 1] * x[data[k]] + data[k + 3] * x[data[k + 2]]) +
            (data[k + 5] * x[data[k + 4]] + data[k + 7] * x[data[k + 6]]);
        }
      }
      for (; k < up; k += 2) {
        int    ind = data[k];
        double val = data[k + 1];
        cp += val * x[ind];
      }

    }

    return cp;
  }

  template<class C>
  C& incr_mult_col(int j, C& x, double coef) {

    if (compact) {

      size_t lo = p[j];
      size_t up = p[j + 1];
      int i = first_i[j];

#ifdef SFBM_HAVE_AVX2_PATH
      if (__builtin_cpu_supports("avx512f")) {
        sfbm_axpy_avx512(data, lo, up, &x[0], i, coef);
        return x;
      }
      if (__builtin_cpu_supports("avx2")) {
        sfbm_axpy_avx2(data, lo, up, &x[0], i, coef);
        return x;
      }
#endif

      size_t k = lo;
      // 8-wide unroll: one prefetch per 64B cache line; per-element FP op
      // order is unchanged (each x[i] is touched exactly once, in order).
      for (; k + 8 <= up; k += 8, i += 8) {
        __builtin_prefetch(&data[k + 64],  0, 1);  // 512B ahead; hint only
        __builtin_prefetch(&data[k + 128], 0, 1);  // 1KB ahead: +MLP
        __builtin_prefetch(&data[k + 192], 0, 1);  // 1.5KB ahead: +MLP
        x[i]     += data[k]     * coef;
        x[i + 1] += data[k + 1] * coef;
        x[i + 2] += data[k + 2] * coef;
        x[i + 3] += data[k + 3] * coef;
        x[i + 4] += data[k + 4] * coef;
        x[i + 5] += data[k + 5] * coef;
        x[i + 6] += data[k + 6] * coef;
        x[i + 7] += data[k + 7] * coef;
      }
      for (; k < up; k++, i++) {
        __builtin_prefetch(&data[k + 64], 0, 1);
        x[i] += data[k] * coef;
      }

    } else {

      size_t lo = 2 * p[j];
      size_t up = 2 * p[j + 1];

      for (size_t k = lo; k < up; k += 2) {
        int    ind = data[k];
        double val = data[k + 1];
        x[ind] += val * coef;
      }

    }

    return x;
  }

protected:
  mio::mmap_source ro_mmap;
  const double * data;
  int n;
  int m;
  std::vector<size_t> p;
  std::vector<int> first_i;
  bool compact;
};

/******************************************************************************/

#endif // SFBM_H
