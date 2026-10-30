# C++ optimization hints

Use these hints when optimizing C++ implementation code, STL-heavy code, numeric kernels, bindings, command-line tools, and compiled libraries. Apply only when profiling indicates the code path matters and behavior is preserved.

1. Optimize by passing large objects by `const&` instead of by value.
2. Optimize by avoiding unnecessary copies of vectors, strings, matrices, and structs.
3. Optimize by using move semantics when transferring ownership of large objects.
4. Optimize by reserving vector or string capacity before repeated `push_back`, `emplace_back`, or append.
5. Optimize by using `emplace_back` when constructing objects in place.
6. Optimize by avoiding repeated heap allocations inside hot loops.
7. Optimize by reusing buffers instead of repeatedly allocating and freeing memory.
8. Optimize by using contiguous memory layouts such as `std::vector`, `std::array`, spans, or flat buffers instead of pointer-heavy containers.
9. Optimize by avoiding `std::list`, `std::map`, and deeply nested containers when vector, flat-map, sorted-vector, or `unordered_map` layouts are more cache-friendly.
10. Optimize by choosing `unordered_map` instead of `map` when ordering is not required.
11. Optimize by using flat arrays, struct-of-arrays layouts, or tightly packed structs for numeric hot paths.
12. Optimize by improving cache locality in nested loops.
13. Optimize by iterating in memory-contiguous order for matrices and arrays.
14. Optimize by avoiding virtual function calls in tight loops when static dispatch, templates, variants, or function objects are appropriate.
15. Optimize by avoiding `std::function` overhead in hot paths when templates, lambdas with `auto` parameters, or function pointers suffice.
16. Optimize by avoiding exceptions for normal control flow.
17. Optimize by avoiding excessive `shared_ptr` use when `unique_ptr`, references, values, or clear ownership are sufficient.
18. Optimize by avoiding atomics, mutexes, locks, and synchronization inside inner loops unless required.
19. Optimize by reducing branch mispredictions in performance-critical code.
20. Optimize by hoisting loop-invariant calculations outside loops.
21. Optimize by avoiding repeated bounds checks, virtual checks, and redundant validation after inputs are already validated.
22. Optimize by using references, iterators, cached indices, or local variables instead of repeated expensive lookups.
23. Optimize by minimizing string parsing, regex, locale-aware formatting, and stream formatting inside hot loops.
24. Optimize by disabling iostream synchronization and avoiding `endl` flushing when using heavy `cin` or `cout`.
25. Optimize by using faster IO strategies for large files, including buffered IO, memory mapping, binary formats, chunked readers, and batched parsers.
26. Optimize by writing SIMD-friendly loops and checking compiler vectorization reports.
27. Optimize by using BLAS, LAPACK, Eigen, Armadillo, oneDNN, cuBLAS, MKL, OpenBLAS, or vendor libraries for dense linear algebra.
28. Optimize by using sparse matrix libraries and sparse-aware algorithms for sparse problems instead of dense representations.
29. Optimize by parallelizing independent loops with OpenMP, TBB, `std::execution`, CUDA, HIP, SYCL, or task schedulers when thread safety and determinism are preserved.
30. Optimize by compiling with appropriate optimization flags and validating improvements with release builds, `perf`, VTune, Callgrind, Massif, `heaptrack`, sanitizer builds, and compiler reports.
