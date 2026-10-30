# Rust optimization hints

Use these hints when optimizing Rust implementation code, Cargo projects, native libraries, CLI tools, parsers, data pipelines, and Rust-backed Python/R extensions. Apply only when profiling indicates the code path matters and behavior is preserved.

1. Optimize by measuring with release builds, because debug builds can be dramatically slower.
2. Optimize by avoiding unnecessary clones of `String`, `Vec`, `HashMap`, structs, and large enum values.
3. Optimize by borrowing with `&T`, `&str`, and slices instead of taking ownership when ownership is not required.
4. Optimize by returning borrowed views or iterators when materializing owned collections is unnecessary and API behavior allows it.
5. Optimize by preallocating `Vec`, `String`, and `HashMap` capacity with `with_capacity` or `reserve`.
6. Optimize by reusing buffers instead of allocating new `Vec` or `String` values inside hot loops.
7. Optimize by avoiding repeated `collect()` calls when iterator chains can remain lazy.
8. Optimize by avoiding long iterator chains in proven hot paths when a simple loop enables better optimization or fewer temporaries.
9. Optimize by using slices and indexed loops when they are clearer and faster for tight numeric kernels.
10. Optimize by removing bounds checks only when safe alternatives, iterators, chunking, or carefully audited `unsafe` code are justified by profiling.
11. Optimize by using `copy_from_slice`, `clone_from_slice`, `extend_from_slice`, or bulk operations instead of element-by-element loops.
12. Optimize by using `HashSet`, `HashMap`, `BTreeMap`, `IndexMap`, sorted vectors, bitsets, or direct indexing according to actual access patterns.
13. Optimize by choosing a faster hasher such as `ahash`, `fxhash`, or `rustc_hash` only when hash security is not required and behavior remains acceptable.
14. Optimize by using `SmallVec`, `ArrayVec`, or stack arrays for small collections that allocate frequently.
15. Optimize by using `Cow` to avoid copies when data is often borrowed but occasionally owned.
16. Optimize by avoiding boxing, trait objects, and dynamic dispatch in hot paths when generics or enums are appropriate.
17. Optimize by avoiding `Arc`, `Rc`, `Mutex`, `RwLock`, atomics, and channels in inner loops unless shared ownership or synchronization is required.
18. Optimize by reducing lock contention through sharding, batching, thread-local buffers, or reducing shared mutable state.
19. Optimize by using Rayon for independent data-parallel work when the work is coarse enough to overcome scheduling overhead.
20. Optimize by avoiding unnecessary UTF-8 validation, string allocation, regex compilation, or formatting in hot paths.
21. Optimize by compiling regexes once with `LazyLock`, `once_cell`, or equivalent caching when regex behavior is required.
22. Optimize by using buffered IO, memory mapping, byte slices, or streaming parsers for large files.
23. Optimize by using `bytes`, `memchr`, `bstr`, `nom`, `csv`, `simd-json`, or domain parsers for parsing-heavy workloads where appropriate.
24. Optimize by representing repeated small categorical values as integers, enums, or interned symbols instead of repeated strings.
25. Optimize by improving data layout with contiguous vectors, struct-of-arrays layouts, packed flags, or bitsets when memory access dominates.
26. Optimize by hoisting invariant computations outside loops.
27. Optimize by avoiding panics, formatting, allocation, and error-construction work on the common fast path.
28. Optimize by using optimized numeric crates such as `ndarray`, `nalgebra`, BLAS bindings, `wide`, `packed_simd` alternatives, or vendor libraries when appropriate.
29. Optimize by inspecting generated code with `cargo asm`, `cargo llvm-lines`, Criterion, flamegraphs, `perf`, and allocation profilers before and after changes.
30. Optimize by keeping safe Rust as the default and using `unsafe` only for small, audited, benchmark-proven sections with explicit invariants and correctness tests.
31. Optimize by tuning the release profile in `Cargo.toml`: `lto = "fat"` or `"thin"`, `codegen-units = 1`, and `panic = "abort"` (when unwinding is not part of behavior) each recover measurable percent on hot binaries.
32. Optimize by building with `RUSTFLAGS="-C target-cpu=native"` (or a specific target CPU) when the deployment machine is known, enabling SIMD codegen the default baseline target forbids.
33. Optimize by swapping the global allocator to `mimalloc` or `jemalloc` when allocation profiling shows malloc dominates; measure first, because the default allocator sometimes wins.
34. Optimize by using profile-guided optimization (`cargo pgo` or `-Cprofile-generate`/`-Cprofile-use`) for mature hot binaries when a representative training workload exists.
35. Optimize by marking tiny hot functions `#[inline]` (or `#[inline(always)]` only with benchmark evidence) and by keeping generic hot code in the same crate as its callers when LTO is off.
36. Optimize by minimizing async overhead: avoid spawning thousands of tiny tasks, batch work per task, prefer `tokio::task::spawn_blocking` for CPU-bound kernels, and poll-ready primitives (`ready!`) over nested async fns in hot paths.
37. Optimize by minimizing FFI boundary costs: batch calls across the boundary, pass slices instead of per-element calls, and avoid allocating across the FFI line.
