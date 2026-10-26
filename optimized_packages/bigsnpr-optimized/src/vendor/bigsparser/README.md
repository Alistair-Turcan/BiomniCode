# Bundled bigsparser header

`SFBM.h` originates from bigsparser 0.7.3, by Florian Privé,
licensed GPL-3: https://github.com/privefl/bigsparser.

This copy contains the accepted compact-column AXPY optimization:
runtime AVX-512F / AVX2 dispatch, scalar fallback, and three-stream
prefetching. It is byte-identical to the header used for evaluation.
The installed bigsparser R package remains a required dependency.
