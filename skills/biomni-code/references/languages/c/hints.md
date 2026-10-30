# C optimization hints

Use these hints when optimizing C implementation code, libc-heavy programs, pointer-heavy code, native extensions, numeric kernels, parsers, and systems code. Apply only when profiling indicates the code path matters and behavior is preserved.

1. Optimize by avoiding unnecessary copies of large structs and buffers.
2. Optimize by passing large structs through pointers to `const` data when mutation is not needed.
3. Optimize by reusing allocated buffers instead of repeatedly calling `malloc` and `free` in hot paths.
4. Optimize by allocating memory in larger chunks or arenas when many small allocations dominate runtime.
5. Optimize by avoiding repeated `realloc` growth and using explicit capacity-doubling strategies.
6. Optimize by storing array lengths and capacities to avoid repeated scans or recomputation.
7. Optimize by using contiguous arrays instead of pointer-heavy linked lists when traversal speed matters.
8. Optimize by improving cache locality with compact structs, flat arrays, and struct-of-arrays layouts where appropriate.
9. Optimize by iterating in memory-contiguous order for matrices and multidimensional arrays.
10. Optimize by moving loop-invariant calculations outside loops.
11. Optimize by reducing function-call overhead in tiny hot functions through `static inline` where appropriate.
12. Optimize by avoiding expensive libc calls such as `strlen`, `strcmp`, `strchr`, `sprintf`, or `sscanf` inside inner loops when cheaper state is available.
13. Optimize by caching string lengths, parsed tokens, hash values, or validated metadata when reused.
14. Optimize by replacing repeated linear searches with hash tables, sorted arrays plus binary search, bitsets, or direct indexing when behavior is preserved.
15. Optimize by avoiding unnecessary zero-initialization of large buffers when every byte will be overwritten before use.
16. Optimize by using `restrict` only when aliasing guarantees are correct and it helps vectorization.
17. Optimize by making pointer aliasing clearer to the compiler so it can safely optimize and vectorize.
18. Optimize by reducing unpredictable branches in hot loops.
19. Optimize by separating rare error handling from the common fast path.
20. Optimize by using buffered IO, `fread`, `fwrite`, memory mapping, or chunked parsers instead of byte-at-a-time IO.
21. Optimize by avoiding repeated `printf`/`fprintf` formatting inside hot loops.
22. Optimize by using binary formats or faster parsers when output compatibility permits.
23. Optimize by using bitsets, packed flags, or lookup tables for repeated boolean or classification logic.
24. Optimize by precomputing lookup tables for repeated deterministic transformations.
25. Optimize by aligning data only when profiling or vectorization evidence shows it matters.
26. Optimize by writing compiler-vectorizable loops and checking vectorization reports.
27. Optimize by using BLAS, LAPACK, FFTW, vendor math libraries, or domain-specific C libraries for dense numerical kernels.
28. Optimize by using sparse-aware data structures and algorithms for sparse problems instead of dense arrays.
29. Optimize by parallelizing independent work with OpenMP, pthreads, thread pools, SIMD, CUDA, HIP, or other native acceleration only when determinism and thread safety are preserved.
30. Optimize by compiling with appropriate optimization flags and validating changes with release builds, `perf`, Callgrind, Massif, `heaptrack`, sanitizers, and hardware counters.
