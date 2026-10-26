#ifndef FASTP_BYTE_MISMATCH_H
#define FASTP_BYTE_MISMATCH_H
#include <cstdint>
#include <cstring>

inline int boundedByteMismatches(const char* a, const char* b, int length, int limit) {
    int count = 0;
    int i = 0;
    for(; i <= length-8; i+=8) {
        uint64_t left, right;
        std::memcpy(&left, a+i, sizeof(left));
        std::memcpy(&right, b+i, sizeof(right));
        uint64_t lanes = left ^ right;
        lanes |= lanes >> 4;
        lanes |= lanes >> 2;
        lanes |= lanes >> 1;
        lanes &= UINT64_C(0x0101010101010101);
        count += (lanes * UINT64_C(0x0101010101010101)) >> 56;
        if(count > limit)
            return count;
    }
    for(; i<length; i++) {
        count += a[i] != b[i];
        if(count > limit)
            return count;
    }
    return count;
}
#endif
