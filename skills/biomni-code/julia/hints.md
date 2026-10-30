# Julia optimization hints

Use these hints when optimizing Julia implementation code, scientific computing packages, numerical kernels, data pipelines, and Julia-backed libraries. Apply only when profiling indicates the code path matters and behavior is preserved.

1. Optimize by measuring with `@time`, `@btime` (BenchmarkTools), `Profile`/`PProf`, and allocation tracking (`@time` reports allocations; use `--track-allocation=user` or AllocationProfiler) before and after each change.
2. Optimize for type stability first: check hot functions with `@code_warntype` and eliminate red (union/abstract) types; type-unstable code defeats specialization.
3. Optimize by declaring concrete field types in structs; abstract or untyped fields force boxing and dynamic dispatch on every access.
4. Optimize by avoiding global variables in hot paths; pass values as arguments or mark them `const`.
5. Optimize by reducing allocations: preallocate output arrays and reuse them (in-place `f!(out, ...)` style) instead of allocating per call.
6. Optimize by using views (`@views`, `view(A, ...)`) instead of slices; slicing copies, views alias.
7. Optimize by fusing broadcasts with dots (`@.`, `a .+ b .* c`) so element-wise expressions run in one loop without temporaries.
8. Optimize by replacing repeated `push!` growth with preallocated arrays when the final size is known or estimable.
9. Optimize by iterating arrays column-major (columns in the inner loop) to match Julia's memory layout.
10. Optimize by using `@inbounds` in proven hot loops after correctness is established, and `@simd` where iterations are independent.
11. Optimize by avoiding abstractly typed containers (`Vector{Any}`, `Dict{Any,Any}`) in hot paths; concrete element types enable unboxed storage.
12. Optimize by using `StaticArrays` for small fixed-size vectors and matrices; stack allocation removes GC pressure entirely.
13. Optimize by choosing `Tuple`/`NamedTuple` over `Vector`/`Dict` for small heterogeneous fixed structures.
14. Optimize by avoiding string operations in numeric kernels; intern repeated categories as integers, enums, or `Symbol`s where behavior allows.
15. Optimize by hoisting invariant computation, regex compilation, and parser construction out of loops.
16. Optimize by using `eachindex`, `axes`, and `CartesianIndices` for generic fast indexing instead of hard-coded `1:length` patterns.
17. Optimize by replacing `findall`-then-index patterns with boolean masks or direct loops when allocation of index vectors dominates.
18. Optimize by using sparse, structured, or lazy matrix types (SparseArrays, LinearAlgebra wrappers, LazyArrays) instead of densifying.
19. Optimize by calling BLAS-backed operations (`mul!`, `axpy!`, broadcasting) instead of hand-written loops for dense linear algebra.
20. Optimize by considering LoopVectorization (`@turbo`) or KernelAbstractions for hot numeric loops when dependencies permit.
21. Optimize by reducing time-to-first-execution with precompilation (PrecompileTools workloads) when startup latency, not steady-state, is the bottleneck.
22. Optimize by avoiding type piracy and excessive specialization: enormous method tables and over-specialized signatures inflate compile time.
23. Optimize by parallelizing with `Threads.@threads` for coarse independent work, FLoops/Transducers for composable parallelism, and Distributed only when data outgrows one machine.
24. Optimize by avoiding race-prone shared writes in threaded loops; use per-thread accumulators and combine.
25. Optimize by batching IO and using buffered readers/writers; use memory-mapped files or Arrow/Parquet for repeated large-data access.
26. Optimize by choosing faster serialization (Arrow, Parquet, JLD2, BSON) appropriate to the data layout when (de)serialization dominates and formats are contractual.
27. Optimize by checking for accidental closures capturing boxed variables (`@code_warntype` on closures); restructure with `let` blocks or explicit arguments.
28. Optimize by avoiding `Dict` lookups with `String` keys in hot paths when small key sets allow vectors, tuples, or `OrderedCollections` alternatives.
29. Optimize by keeping hot loops free of IO, logging, and plotting calls; batch diagnostics outside the kernel.
30. Optimize by reading the allocation line of `@btime` before the time line: in idiomatic Julia, near-zero allocations in the hot loop is the strongest single predictor of speed.
