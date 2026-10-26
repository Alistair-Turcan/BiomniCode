#include <stdint.h>
#include <string.h>
#include <stdio.h>
#define NDEBUG /* r3i017: assert conditions are side-effect-free pure checks */
#include <assert.h>
#include "mmpriv.h"
#include "kalloc.h"
#include "krmq.h"
#if defined(__SSE2__) && (defined(__GNUC__) || defined(__clang__))
#define MM_DP_MASK16 1
#ifdef USE_SIMDE
#include <simde/x86/sse2.h>
#else
#include <emmintrin.h>
#endif
#if defined(__x86_64__) && !defined(USE_SIMDE)
#define MM_DP_MASK_AVX2 1
#define MM_DP_MASK_AVX512 1
#include <immintrin.h>
#endif
#endif

static int64_t mg_chain_bk_end(int32_t max_drop, const mm128_t *z, const int32_t *f, const int64_t *p, int32_t *t, int64_t k)
{
	int64_t i = z[k].y, end_i = -1, max_i = i;
	int32_t max_s = 0;
	if (i < 0 || t[i] != 0) return i;
	do {
		int32_t s;
		t[i] = 2;
		end_i = i = p[i];
		s = i < 0? z[k].x : (int32_t)z[k].x - f[i];
		if (s > max_s) max_s = s, max_i = i;
		else if (max_s - s > max_drop) break;
	} while (i >= 0 && t[i] == 0);
	for (i = max_i; i >= 0 && i != end_i; i = p[i]) // reset only the tail the caller does not overwrite with t=1
		t[i] = 0;
	return max_i;
}

uint64_t *mg_chain_backtrack(void *km, int64_t n, const int32_t *f, const int64_t *p, int32_t *v, int32_t *t, int32_t min_cnt, int32_t min_sc, int32_t max_drop, int32_t *n_u_, int32_t *n_v_)
{
	mm128_t *z;
	uint64_t *u;
	int64_t i, k, n_z, n_v;
	int32_t n_u;

	*n_u_ = *n_v_ = 0;
	z = Kmalloc(km, mm128_t, n); /* over-allocate to n: fill in one pass, n_z found on the fly */
	for (i = 0, n_z = 0; i < n; ++i)
		if (f[i] >= min_sc) z[n_z].x = f[i], z[n_z].y = i, ++n_z;
	if (n_z == 0) { kfree(km, z); return 0; }
	radix_sort_128x(z, z + n_z);

	/* r3i015: single fused pass. The original did a count-only pass followed by
	 * an identical populate pass; the two walks see exactly the same t[] states
	 * and make the same accept/reject decisions, so fusing them (with u[]
	 * over-allocated to n_z >= n_u) produces identical u[]/v[]/n_u/n_v while
	 * halving the mg_chain_bk_end calls and t[] memsets. */
	u = Kmalloc(km, uint64_t, n_z);
	memset(t, 0, n * 4);
	for (k = n_z - 1, n_v = n_u = 0; k >= 0; --k) { // populate u[] and count in one pass
		if (t[z[k].y] == 0) {
			int64_t n_v0 = n_v, end_i;
			int32_t sc;
			end_i = mg_chain_bk_end(max_drop, z, f, p, t, k);
			for (i = z[k].y; i != end_i; i = p[i])
				v[n_v++] = i, t[i] = 1;
			sc = i < 0? z[k].x : (int32_t)z[k].x - f[i];
			if (sc >= min_sc && n_v > n_v0 && n_v - n_v0 >= min_cnt)
				u[n_u++] = (uint64_t)sc << 32 | (n_v - n_v0);
			else n_v = n_v0;
		}
	}
	kfree(km, z);
	assert(n_v < INT32_MAX);
	*n_u_ = n_u, *n_v_ = n_v;
	return u;
}

static mm128_t *compact_a(void *km, int32_t n_u, uint64_t *u, int32_t n_v, int32_t *v, mm128_t *a)
{
	mm128_t *b, *w;
	uint64_t *u2;
	int64_t i, j, k;

	// write the result to b[]
	b = Kmalloc(km, mm128_t, n_v);
	for (i = 0, k = 0; i < n_u; ++i) {
		int32_t k0 = k, ni = (int32_t)u[i];
		for (j = 0; j < ni; ++j)
			b[k++] = a[v[k0 + (ni - j - 1)]];
	}
	kfree(km, v);

	// sort u[] and a[] by the target position, such that adjacent chains may be joined
	w = Kmalloc(km, mm128_t, n_u);
	for (i = k = 0; i < n_u; ++i) {
		w[i].x = b[k].x, w[i].y = (uint64_t)k<<32|i;
		k += (int32_t)u[i];
	}
	radix_sort_128x(w, w + n_u);
	u2 = Kmalloc(km, uint64_t, n_u);
	for (i = k = 0; i < n_u; ++i) {
		int32_t j = (int32_t)w[i].y, n = (int32_t)u[j];
		u2[i] = u[j];
		memcpy(&a[k], &b[w[i].y>>32], n * sizeof(mm128_t));
		k += n;
	}
	memcpy(u, u2, n_u * 8);
	memcpy(b, a, k * sizeof(mm128_t)); // write _a_ to _b_ and deallocate _a_ because _a_ is oversized, sometimes a lot
	kfree(km, a); kfree(km, w); kfree(km, u2);
	return b;
}

static inline
#if defined(__GNUC__) || defined(__clang__)
__attribute__((always_inline))
#endif
int32_t comput_sc(const mm128_t *ai, const mm128_t *aj, int32_t max_dist_x, int32_t max_dist_y, int32_t bw, float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg, int query_validated)
{
	int32_t dq = (int32_t)ai->y - (int32_t)aj->y, dr, dd, dg, q_span, sc;
	int32_t sidi = (ai->y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
	int32_t sidj = (aj->y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
	if (!query_validated && (max_dist_x <= 0 || (uint32_t)dq - 1U >= (uint32_t)max_dist_x)) return INT32_MIN;
	dr = (int32_t)(ai->x - aj->x);
	if (sidi == sidj && (dr == 0 || dq > max_dist_y)) return INT32_MIN;
	dd = dr > dq? dr - dq : dq - dr;
	if (sidi == sidj && dd > bw) return INT32_MIN;
	if (n_seg > 1 && !is_cdna && sidi == sidj && dr > max_dist_y) return INT32_MIN;
	dg = dr < dq? dr : dq;
	q_span = aj->y>>32&0xff;
	sc = q_span < dg? q_span : dg;
	if (dd || dg > q_span) {
		float lin_pen, log_pen;
		lin_pen = chn_pen_gap * (float)dd + chn_pen_skip * (float)dg;
		log_pen = dd >= 1? mg_log2(dd + 1) : 0.0f; // mg_log2() only works for dd>=2
		if (is_cdna || sidi != sidj) {
			if (sidi != sidj && dr == 0) ++sc; // possibly due to overlapping paired ends; give a minor bonus
			else if (dr > dq || sidi != sidj) sc -= (int)(lin_pen < log_pen? lin_pen : log_pen); // deletion or jump between paired ends
			else sc -= (int)(lin_pen + .5f * log_pen);
		} else sc -= (int)(lin_pen + .5f * log_pen);
	}
	return sc;
}

#ifdef MM_DP_MASK_AVX512
/* AVX-512F-target per-i scan: 64-lane blocks (half the block iterations of the
 * AVX2 version) computing only the 16-lane groups intersecting [bottom, j]
 * (r3i015 lane-group skipping lifted to 512-bit). Predicates are bit-identical
 * to the AVX2/SSE masks: dq in [1,lim] <=> (uint32)(dq-1)<(uint32)lim;
 * |dd|<=bw <=> (uint32)(dd+bw)<(uint32)(2bw+1). Set bits are visited in
 * strictly descending j, so comput_sc order and t[] updates are identical. */
__attribute__((target("avx512f")))
static int64_t dp_scan_avx512(const mm128_t *a, const int32_t *q_pos, const int32_t *x_pos, const int32_t *f, int32_t *t, const int64_t *p,
							int64_t i, int64_t st, int32_t max_dist_x, int32_t max_dist_y, int32_t bw,
							float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg,
							int32_t lim, int full, int32_t max_skip, int32_t *max_f_io, int64_t *max_j_io, int32_t *n_skip_io)
{
	int64_t j, max_j = *max_j_io;
	int32_t max_f = *max_f_io, n_skip = *n_skip_io;
	const __m512i qv = _mm512_set1_epi32(q_pos[i]), xv = _mm512_set1_epi32(x_pos[i]);
	const __m512i onev = _mm512_set1_epi32(1);
	const __m512i limv = _mm512_set1_epi32(lim), bwv = _mm512_set1_epi32(bw), bw2 = _mm512_set1_epi32(2 * bw + 1);
	for (j = i - 1; j >= st; ) {
		int64_t base = (j >> 6) << 6, bottom = base < st? st : base;
		uint64_t mask = 0;
		int lane, ltop = (int)(j - base), lane0 = (int)(bottom - base) & ~15;
		for (lane = lane0; lane <= ltop; lane += 16) {
			__m512i dq = _mm512_sub_epi32(qv, _mm512_loadu_si512((const void*)(q_pos + base + lane)));
			__mmask16 valid = _mm512_cmplt_epu32_mask(_mm512_sub_epi32(dq, onev), limv); /* dq in [1, lim] */
			if (full) {
				__m512i dr = _mm512_sub_epi32(xv, _mm512_loadu_si512((const void*)(x_pos + base + lane)));
				__m512i ddb = _mm512_add_epi32(_mm512_sub_epi32(dr, dq), bwv); /* dd + bw */
				valid = valid & _mm512_cmplt_epu32_mask(ddb, bw2); /* |dd| <= bw */
			}
			mask |= (uint64_t)valid << lane;
		}
		mask &= (~0ULL >> (63 - (j - base))) & (~0ULL << (bottom - base));
		while (mask) {
			unsigned bit = 63U - __builtin_clzll(mask);
			int32_t sc;
			j = base + bit;
			mask &= ~(1ULL << bit);
			sc = comput_sc(&a[i], &a[j], max_dist_x, max_dist_y, bw, chn_pen_gap, chn_pen_skip, is_cdna, n_seg, 1);
			if (sc == INT32_MIN) continue;
			sc += f[j];
			if (sc > max_f) {
				max_f = sc, max_j = j;
				if (n_skip > 0) --n_skip;
			} else if (t[j] == (int32_t)i) {
				if (++n_skip > max_skip) { *max_f_io = max_f; *max_j_io = max_j; *n_skip_io = n_skip; return j; }
			}
			if (p[j] >= 0) t[p[j]] = i;
		}
		j = bottom - 1;
	}
	*max_f_io = max_f; *max_j_io = max_j; *n_skip_io = n_skip;
	return j;
}
#endif

#ifdef MM_DP_MASK_AVX2
/* AVX2-target per-i scan: dp_mask32 inlines into this caller, removing the
 * per-block call overhead (~82M calls). Performs exactly the operations of the
 * have_avx2 branch of the mg_lchain_dp j-loop, in the same order and with the
 * same values -> byte-identical. Returns the ending j (end_j semantics). */
__attribute__((target("avx2")))
static int64_t dp_scan_avx2(const mm128_t *a, const int32_t *q_pos, const int32_t *x_pos, const int32_t *f, int32_t *t, const int64_t *p,
							int64_t i, int64_t st, int32_t max_dist_x, int32_t max_dist_y, int32_t bw,
							float chn_pen_gap, float chn_pen_skip, int is_cdna, int n_seg,
							int32_t lim, int full, int32_t max_skip, int32_t *max_f_io, int64_t *max_j_io, int32_t *n_skip_io)
{
	int64_t j, max_j = *max_j_io;
	int32_t max_f = *max_f_io, n_skip = *n_skip_io;
	/* hoist all loop-invariant broadcasts out of the per-block mask computation
	 * (identical lane values -> byte-identical; removes ~7 broadcasts per block) */
	const __m256i qv = _mm256_set1_epi32(q_pos[i]), xv = _mm256_set1_epi32(x_pos[i]);
	const __m256i onev = _mm256_set1_epi32(1), sign = _mm256_set1_epi32(INT32_MIN);
	const __m256i limv = _mm256_set1_epi32(lim), bwv = _mm256_set1_epi32(bw), bw2 = _mm256_set1_epi32(2 * bw + 1);
	for (j = i - 1; j >= st; ) {
		int64_t base = (j >> 5) << 5, bottom = base < st? st : base;
		uint64_t mask = 0;
		/* r3i015: only compute 8-lane groups intersecting [bottom, j]; groups
		 * entirely outside are zeroed by the refinement below anyway, so the
		 * resulting mask (and all downstream state) is bit-identical. */
		int lane, ltop = (int)(j - base), lane0 = (int)(bottom - base) & ~7;
		for (lane = lane0; lane <= ltop; lane += 8) {
			__m256i dq = _mm256_sub_epi32(qv, _mm256_loadu_si256((const __m256i*)(q_pos + base + lane)));
			__m256i valid = _mm256_cmpgt_epi32(_mm256_xor_si256(limv, sign), _mm256_xor_si256(_mm256_sub_epi32(dq, onev), sign)); /* dq in [1, lim] */
			if (full) {
				__m256i dr = _mm256_sub_epi32(xv, _mm256_loadu_si256((const __m256i*)(x_pos + base + lane)));
				__m256i ddb = _mm256_add_epi32(_mm256_sub_epi32(dr, dq), bwv); /* dd + bw */
				valid = _mm256_and_si256(valid, _mm256_cmpgt_epi32(_mm256_xor_si256(bw2, sign), _mm256_xor_si256(ddb, sign))); /* |dd| <= bw */
			}
			mask |= (uint64_t)_mm256_movemask_ps(_mm256_castsi256_ps(valid)) << lane;
		}
		mask &= ((1ULL << (j - base + 1)) - 1) & (~0ULL << (bottom - base));
		while (mask) {
			unsigned bit = 63U - __builtin_clzll(mask);
			int32_t sc;
			j = base + bit;
			mask &= ~(1ULL << bit);
			sc = comput_sc(&a[i], &a[j], max_dist_x, max_dist_y, bw, chn_pen_gap, chn_pen_skip, is_cdna, n_seg, 1);
			if (sc == INT32_MIN) continue;
			sc += f[j];
			if (sc > max_f) {
				max_f = sc, max_j = j;
				if (n_skip > 0) --n_skip;
			} else if (t[j] == (int32_t)i) {
				if (++n_skip > max_skip) { *max_f_io = max_f; *max_j_io = max_j; *n_skip_io = n_skip; return j; }
			}
			if (p[j] >= 0) t[p[j]] = i;
		}
		j = bottom - 1;
	}
	*max_f_io = max_f; *max_j_io = max_j; *n_skip_io = n_skip;
	return j;
}
#endif

/* Input:
 *   a[].x: rev<<63 | tid<<32 | tpos
 *   a[].y: flags<<40 | q_span<<32 | q_pos
 * Output:
 *   n_u: #chains
 *   u[]: score<<32 | #anchors (sum of lower 32 bits of u[] is the returned length of a[])
 * input a[] is deallocated on return
 */
mm128_t *mg_lchain_dp(int max_dist_x, int max_dist_y, int bw, int max_skip, int max_iter, int min_cnt, int min_sc, float chn_pen_gap, float chn_pen_skip,
					  int is_cdna, int n_seg, int64_t n, mm128_t *a, int *n_u_, uint64_t **_u, void *km)
{ // TODO: make sure this works when n has more than 32 bits
	int32_t *f, *t, *v, n_u, n_v, mmax_f = 0, max_drop = bw;
	int64_t *p, i, j, max_ii, st = 0;
	uint64_t *u;
#ifdef MM_DP_MASK16
	int32_t *q_pos, *x_pos, *q_pos_raw, *x_pos_raw;
	int32_t lim_xy, dp_mask_full;
#ifdef MM_DP_MASK_AVX2
	int have_avx2;
#endif
#ifdef MM_DP_MASK_AVX512
	int have_avx512;
#endif
#endif

	if (_u) *_u = 0, *n_u_ = 0;
	if (n == 0 || a == 0) {
		kfree(km, a);
		return 0;
	}
	if (max_dist_x < bw) max_dist_x = bw;
	if (max_dist_y < bw && !is_cdna) max_dist_y = bw;
	if (is_cdna) max_drop = INT32_MAX;
	p = Kmalloc(km, int64_t, n);
	f = Kmalloc(km, int32_t, n);
	v = Kmalloc(km, int32_t, n);
	t = Kcalloc(km, int32_t, n);
#ifdef MM_DP_MASK16
	q_pos_raw = Kcalloc(km, int32_t, ((n + 63) / 64) * 64 + 16);
	x_pos_raw = Kcalloc(km, int32_t, ((n + 63) / 64) * 64 + 16);
	q_pos = (int32_t*)(((uintptr_t)q_pos_raw + 63) & ~(uintptr_t)63); /* 64B-aligned: 512-bit loads never split lines */
	x_pos = (int32_t*)(((uintptr_t)x_pos_raw + 63) & ~(uintptr_t)63);
	for (i = 0; i < n; ++i) q_pos[i] = (int32_t)a[i].y, x_pos[i] = (int32_t)a[i].x;
	dp_mask_full = (n_seg == 1); /* single segment: sidi==sidj, so comput_sc rejects exactly dr==0, dq>max_dist_y, dd>bw */
	lim_xy = max_dist_x < max_dist_y? max_dist_x : max_dist_y;
#ifdef MM_DP_MASK_AVX2
	have_avx2 = __builtin_cpu_supports("avx2");
#ifdef MM_DP_MASK_AVX512
	have_avx512 = __builtin_cpu_supports("avx512f");
#endif
#endif
#endif

	// fill the score and backtrack arrays
	for (i = 0, max_ii = -1; i < n; ++i) {
		int64_t max_j = -1, end_j;
		int32_t max_f = a[i].y>>32&0xff, n_skip = 0;
		while (st < i && (a[i].x>>32 != a[st].x>>32 || a[i].x > a[st].x + max_dist_x)) ++st;
		if (i - st > max_iter) st = i - max_iter;
#ifdef MM_DP_MASK16
#ifdef MM_DP_MASK_AVX2
#ifdef MM_DP_MASK_AVX512
		if (have_avx512) {
			j = dp_scan_avx512(a, q_pos, x_pos, f, t, p, i, st, max_dist_x, max_dist_y, bw,
							   chn_pen_gap, chn_pen_skip, is_cdna, n_seg,
							   dp_mask_full? lim_xy : max_dist_x, dp_mask_full, max_skip,
							   &max_f, &max_j, &n_skip);
		} else
#endif
		if (have_avx2) {
			j = dp_scan_avx2(a, q_pos, x_pos, f, t, p, i, st, max_dist_x, max_dist_y, bw,
							 chn_pen_gap, chn_pen_skip, is_cdna, n_seg,
							 dp_mask_full? lim_xy : max_dist_x, dp_mask_full, max_skip,
							 &max_f, &max_j, &n_skip);
		} else
#endif
		for (j = i - 1; j >= st; ) {
			{
			int64_t base = (j >> 4) << 4, bottom = base < st? st : base;
			unsigned mask = 0, lane;
			__m128i qi = _mm_set1_epi32(q_pos[i]), xi = _mm_set1_epi32(x_pos[i]);
			__m128i onev = _mm_set1_epi32(1), sign = _mm_set1_epi32(INT32_MIN);
			__m128i limx = _mm_set1_epi32(dp_mask_full? lim_xy : max_dist_x);
			__m128i bwv = _mm_set1_epi32(bw), bw2 = _mm_set1_epi32(2 * bw + 1);
			for (lane = 0; lane < 16; lane += 4) {
				__m128i dq = _mm_sub_epi32(qi, _mm_loadu_si128((const __m128i*)(q_pos + base + lane)));
				__m128i valid = _mm_cmplt_epi32(_mm_xor_si128(_mm_sub_epi32(dq, onev), sign), _mm_xor_si128(limx, sign)); /* dq in [1, lim] */
				if (dp_mask_full) {
					__m128i dr = _mm_sub_epi32(xi, _mm_loadu_si128((const __m128i*)(x_pos + base + lane)));
					__m128i ddb = _mm_add_epi32(_mm_sub_epi32(dr, dq), bwv); /* dd + bw */
					valid = _mm_and_si128(valid, _mm_cmplt_epi32(_mm_xor_si128(ddb, sign), _mm_xor_si128(bw2, sign))); /* |dd| <= bw */
				}
				mask |= (unsigned)_mm_movemask_ps(_mm_castsi128_ps(valid)) << lane;
			}
			mask &= ((1U << (j - base + 1)) - 1) & (0xffffU << (bottom - base));
			while (mask) {
				unsigned bit = 31U - __builtin_clz(mask);
				int32_t sc;
				j = base + bit;
				mask &= ~(1U << bit);
				sc = comput_sc(&a[i], &a[j], max_dist_x, max_dist_y, bw, chn_pen_gap, chn_pen_skip, is_cdna, n_seg, 1);
				if (sc == INT32_MIN) continue;
				sc += f[j];
				if (sc > max_f) {
					max_f = sc, max_j = j;
					if (n_skip > 0) --n_skip;
				} else if (t[j] == (int32_t)i) {
					if (++n_skip > max_skip) goto dp_mask_scan_end;
				}
				if (p[j] >= 0) t[p[j]] = i;
			}
			j = bottom - 1;
			}
		}
dp_mask_scan_end:
#else
		for (j = i - 1; j >= st; --j) {
			int32_t sc;
			sc = comput_sc(&a[i], &a[j], max_dist_x, max_dist_y, bw, chn_pen_gap, chn_pen_skip, is_cdna, n_seg, 0);
			if (sc == INT32_MIN) continue;
			sc += f[j];
			if (sc > max_f) {
				max_f = sc, max_j = j;
				if (n_skip > 0) --n_skip;
			} else if (t[j] == (int32_t)i) {
				if (++n_skip > max_skip)
					break;
			}
			if (p[j] >= 0) t[p[j]] = i;
		}
#endif
		end_j = j;
		if (max_ii < 0 || a[i].x - a[max_ii].x > (int64_t)max_dist_x) {
			int32_t max = INT32_MIN;
			max_ii = -1;
			for (j = i - 1; j >= st; --j)
				if (max < f[j]) max = f[j], max_ii = j;
		}
		if (max_ii >= 0 && max_ii < end_j) {
			int32_t tmp;
			tmp = comput_sc(&a[i], &a[max_ii], max_dist_x, max_dist_y, bw, chn_pen_gap, chn_pen_skip, is_cdna, n_seg, 0);
			if (tmp != INT32_MIN && max_f < tmp + f[max_ii])
				max_f = tmp + f[max_ii], max_j = max_ii;
		}
		f[i] = max_f, p[i] = max_j;
		v[i] = max_j >= 0 && v[max_j] > max_f? v[max_j] : max_f; // v[] keeps the peak score up to i; f[] is the score ending at i, not always the peak
		if (max_ii < 0 || (a[i].x - a[max_ii].x <= (int64_t)max_dist_x && f[max_ii] < f[i]))
			max_ii = i;
		if (mmax_f < max_f) mmax_f = max_f;
		//fprintf(stderr, "X1\t%ld\t%ld:%d\t%ld\t%ld:%d\t%ld\t%ld\n", (long)i, (long)(a[i].x>>32), (int32_t)a[i].x, (long)max_j, max_j<0?-1L:(long)(a[max_j].x>>32), max_j<0?-1:(int32_t)a[max_j].x, (long)max_f, (long)v[i]);
	}

	u = mg_chain_backtrack(km, n, f, p, v, t, min_cnt, min_sc, max_drop, &n_u, &n_v);
	*n_u_ = n_u, *_u = u; // NB: note that u[] may not be sorted by score here
	kfree(km, p); kfree(km, f); kfree(km, t);
#ifdef MM_DP_MASK16
	kfree(km, q_pos_raw); kfree(km, x_pos_raw);
#endif
	if (n_u == 0) {
		kfree(km, a); kfree(km, v);
		return 0;
	}
	return compact_a(km, n_u, u, n_v, v, a);
}

typedef struct lc_elem_s {
	int32_t y;
	int64_t i;
	double pri;
	KRMQ_HEAD(struct lc_elem_s) head;
} lc_elem_t;

#define lc_elem_cmp(a, b) ((a)->y < (b)->y? -1 : (a)->y > (b)->y? 1 : ((a)->i > (b)->i) - ((a)->i < (b)->i))
#define lc_elem_lt2(a, b) ((a)->pri < (b)->pri)
KRMQ_INIT(lc_elem, lc_elem_t, head, lc_elem_cmp, lc_elem_lt2)

KALLOC_POOL_INIT(rmq, lc_elem_t)

static inline int32_t comput_sc_simple(const mm128_t *ai, const mm128_t *aj, float chn_pen_gap, float chn_pen_skip, int32_t *exact, int32_t *width)
{
	int32_t dq = (int32_t)ai->y - (int32_t)aj->y, dr, dd, dg, q_span, sc;
	dr = (int32_t)(ai->x - aj->x);
	*width = dd = dr > dq? dr - dq : dq - dr;
	dg = dr < dq? dr : dq;
	q_span = aj->y>>32&0xff;
	sc = q_span < dg? q_span : dg;
	if (exact) *exact = (dd == 0 && dg <= q_span);
	if (dd || dq > q_span) {
		float lin_pen, log_pen;
		lin_pen = chn_pen_gap * (float)dd + chn_pen_skip * (float)dg;
		log_pen = dd >= 1? mg_log2(dd + 1) : 0.0f; // mg_log2() only works for dd>=2
		sc -= (int)(lin_pen + .5f * log_pen);
	}
	return sc;
}

mm128_t *mg_lchain_rmq(int max_dist, int max_dist_inner, int bw, int max_chn_skip, int cap_rmq_size, int min_cnt, int min_sc, float chn_pen_gap, float chn_pen_skip,
					   int64_t n, mm128_t *a, int *n_u_, uint64_t **_u, void *km)
{
	int32_t *f,*t, *v, n_u, n_v, mmax_f = 0, max_rmq_size = 0, max_drop = bw;
	int64_t *p, i, i0, st = 0, st_inner = 0;
	uint64_t *u;
	lc_elem_t *root = 0, *root_inner = 0;
	void *mem_mp = 0;
	kmp_rmq_t *mp;

	if (_u) *_u = 0, *n_u_ = 0;
	if (n == 0 || a == 0) {
		kfree(km, a);
		return 0;
	}
	if (max_dist < bw) max_dist = bw;
	if (max_dist_inner < 0) max_dist_inner = 0;
	if (max_dist_inner > max_dist) max_dist_inner = max_dist;
	p = Kmalloc(km, int64_t, n);
	f = Kmalloc(km, int32_t, n);
	t = Kcalloc(km, int32_t, n);
	v = Kmalloc(km, int32_t, n);
	mem_mp = km_init2(km, 0x10000);
	mp = kmp_init_rmq(mem_mp);

	// fill the score and backtrack arrays
	for (i = i0 = 0; i < n; ++i) {
		int64_t max_j = -1;
		int32_t q_span = a[i].y>>32&0xff, max_f = q_span;
		lc_elem_t s, *q, *r, lo, hi;
		// add in-range anchors
		if (i0 < i && a[i0].x != a[i].x) {
			int64_t j;
			for (j = i0; j < i; ++j) {
				q = kmp_alloc_rmq(mp);
				q->y = (int32_t)a[j].y, q->i = j, q->pri = -(f[j] + 0.5 * chn_pen_gap * ((int32_t)a[j].x + (int32_t)a[j].y));
				krmq_insert(lc_elem, &root, q, 0);
				if (max_dist_inner > 0) {
					r = kmp_alloc_rmq(mp);
					*r = *q;
					krmq_insert(lc_elem, &root_inner, r, 0);
				}
			}
			i0 = i;
		}
		// get rid of active chains out of range
		while (st < i && (a[i].x>>32 != a[st].x>>32 || a[i].x > a[st].x + max_dist || krmq_size(head, root) > cap_rmq_size)) {
			s.y = (int32_t)a[st].y, s.i = st;
			if ((q = krmq_find(lc_elem, root, &s, 0)) != 0) {
				q = krmq_erase(lc_elem, &root, q, 0);
				kmp_free_rmq(mp, q);
			}
			++st;
		}
		if (max_dist_inner > 0)  { // similar to the block above, but applied to the inner tree
			while (st_inner < i && (a[i].x>>32 != a[st_inner].x>>32 || a[i].x > a[st_inner].x + max_dist_inner || krmq_size(head, root_inner) > cap_rmq_size)) {
				s.y = (int32_t)a[st_inner].y, s.i = st_inner;
				if ((q = krmq_find(lc_elem, root_inner, &s, 0)) != 0) {
					q = krmq_erase(lc_elem, &root_inner, q, 0);
					kmp_free_rmq(mp, q);
				}
				++st_inner;
			}
		}
		// RMQ
		lo.i = INT32_MAX, lo.y = (int32_t)a[i].y - max_dist;
		hi.i = 0, hi.y = (int32_t)a[i].y;
		if ((q = krmq_rmq(lc_elem, root, &lo, &hi)) != 0) {
			int32_t sc, exact, width, n_skip = 0;
			int64_t j = q->i;
			assert(q->y >= lo.y && q->y <= hi.y);
			sc = f[j] + comput_sc_simple(&a[i], &a[j], chn_pen_gap, chn_pen_skip, &exact, &width);
			if (width <= bw && sc > max_f) max_f = sc, max_j = j;
			if (!exact && root_inner && (int32_t)a[i].y > 0) {
				lc_elem_t *lo, *hi;
				s.y = (int32_t)a[i].y - 1, s.i = n;
				krmq_interval(lc_elem, root_inner, &s, &lo, &hi);
				if (lo) {
					const lc_elem_t *q;
					int32_t width;
					krmq_itr_t(lc_elem) itr;
					krmq_itr_find(lc_elem, root_inner, lo, &itr);
					while ((q = krmq_at(&itr)) != 0) {
						if (q->y < (int32_t)a[i].y - max_dist_inner) break;
						j = q->i;
						sc = f[j] + comput_sc_simple(&a[i], &a[j], chn_pen_gap, chn_pen_skip, 0, &width);
						if (width <= bw) {
							if (sc > max_f) {
								max_f = sc, max_j = j;
								if (n_skip > 0) --n_skip;
							} else if (t[j] == (int32_t)i) {
								if (++n_skip > max_chn_skip)
									break;
							}
							if (p[j] >= 0) t[p[j]] = i;
						}
						if (!krmq_itr_prev(lc_elem, &itr)) break;
					}
				}
			}
		}
		// set max
		assert(max_j < 0 || (a[max_j].x < a[i].x && (int32_t)a[max_j].y < (int32_t)a[i].y));
		f[i] = max_f, p[i] = max_j;
		v[i] = max_j >= 0 && v[max_j] > max_f? v[max_j] : max_f; // v[] keeps the peak score up to i; f[] is the score ending at i, not always the peak
		if (mmax_f < max_f) mmax_f = max_f;
		if (max_rmq_size < krmq_size(head, root)) max_rmq_size = krmq_size(head, root);
	}
	km_destroy(mem_mp);

	u = mg_chain_backtrack(km, n, f, p, v, t, min_cnt, min_sc, max_drop, &n_u, &n_v);
	*n_u_ = n_u, *_u = u; // NB: note that u[] may not be sorted by score here
	kfree(km, p); kfree(km, f); kfree(km, t);
	if (n_u == 0) {
		kfree(km, a); kfree(km, v);
		return 0;
	}
	return compact_a(km, n_u, u, n_v, v, a);
}
