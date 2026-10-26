/****
DIAMOND protein sequence aligner
Copyright (C) 2026
SPDX-License-Identifier: GPL-3.0-or-later
****/
#pragma once

#if defined(__SSE4_1__) && defined(SEQ_MASK)
#include <smmintrin.h>
#include "basic/shape.h"

namespace DISPATCH_ARCH {

// Only used with base ten, weight eight, positions <16, and 16 readable bytes.
struct ReducedSeed8 {
	explicit ReducedSeed8(const Shape& shape) :
		gather(_mm_setr_epi8(
			shape.positions_[0], shape.positions_[1], shape.positions_[2], shape.positions_[3],
			shape.positions_[4], shape.positions_[5], shape.positions_[6], shape.positions_[7],
			-128, -128, -128, -128, -128, -128, -128, -128))
	{}

	bool get(PackedSeed& seed, const Letter* ptr) const {
		__m128i letters = _mm_shuffle_epi8(
			_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)), gather);
		letters = _mm_and_si128(letters, _mm_set1_epi8(LETTER_MASK));
		if (_mm_movemask_epi8(_mm_cmpeq_epi8(letters, _mm_set1_epi8(MASK_LETTER))) & 255)
			return false;
		const __m128i lo = _mm_cvtepu8_epi32(letters);
		const __m128i hi = _mm_cvtepu8_epi32(_mm_srli_si128(letters, 4));
		__m128i sums = _mm_add_epi32(
			_mm_mullo_epi32(lo, _mm_setr_epi32(10000000, 1000000, 100000, 10000)),
			_mm_mullo_epi32(hi, _mm_setr_epi32(1000, 100, 10, 1)));
		sums = _mm_hadd_epi32(sums, sums);
		sums = _mm_hadd_epi32(sums, sums);
		seed = static_cast<uint32_t>(_mm_cvtsi128_si32(sums));
		return true;
	}

	__m128i gather;
};

}
#endif
