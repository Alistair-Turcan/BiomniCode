#ifndef FASTP_KNOWN_ADAPTER_SEEDS_H
#define FASTP_KNOWN_ADAPTER_SEEDS_H

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

class ReadFourmerIndex {
public:
    static int baseCode(char base) {
        switch(base) {
            case 'A': return 0;
            case 'C': return 1;
            case 'G': return 2;
            case 'T': return 3;
            default: return -1;
        }
    }
    void build(const char* data, int length) {
        heads.fill(-1);
        next.resize(length);
        int key=0, run=0;
        for(int i=length-1; i>=0; i--) {
            const int base=baseCode(data[i]);
            if(base < 0) { key=0; run=0; continue; }
            key=(key>>2) | (base<<6);
            if(run < 4) ++run;
            if(run == 4) {
                next[i]=heads[key];
                heads[key]=i;
            }
        }
    }
    int occurrence(int code, int minimum) const {
        int pos=heads[code];
        while(pos >= 0 && pos < minimum)
            pos=next[pos];
        return pos;
    }
private:
    std::array<int,256> heads;
    std::vector<int> next;
};

class KnownAdapterSeeds {
public:
    explicit KnownAdapterSeeds(const std::string& adapter) {
        const size_t count = adapter.size()/16 + 1;
        if(count * sizeof(uint32_t) > adapter.size())
            return;
        words.resize(count);
        for(size_t i=0; i<count; i++) {
            std::memcpy(&words[i], adapter.data()+i*sizeof(uint32_t), sizeof(uint32_t));
            int code=0;
            for(int j=0; j<4; j++) {
                const int base=ReadFourmerIndex::baseCode(adapter[i*4+j]);
                if(base < 0) { indexed=false; break; }
                code=(code<<2) | base;
            }
            codes.push_back(code);
        }
    }

    int nextStart(const ReadFourmerIndex& index, int minimum, int length) const {
        if(words.empty() || !indexed)
            return minimum;
        const int coveredEnd=length-int(words.size()*sizeof(uint32_t));
        if(minimum > coveredEnd)
            return minimum;
        int best=coveredEnd+1;
        for(size_t i=0; i<codes.size(); i++) {
            const int seedOffset=int(i*sizeof(uint32_t));
            const int pos=index.occurrence(codes[i],minimum+seedOffset);
            if(pos >= 0)
                best=std::min(best,pos-seedOffset);
        }
        return best;
    }

    bool possible(const char* read, int length) const {
        if(length < words.size()*sizeof(uint32_t))
            return true;
        if(words.empty())
            return true;
        for(size_t i=0; i<words.size(); i++) {
            uint32_t word;
            std::memcpy(&word, read+i*sizeof(uint32_t), sizeof(uint32_t));
            if(word == words[i])
                return true;
        }
        return false;
    }

private:
    std::vector<uint32_t> words;
    std::vector<int> codes;
    bool indexed=true;
};
#endif
