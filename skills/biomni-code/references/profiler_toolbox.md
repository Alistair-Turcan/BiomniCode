# Profiler toolbox

Per-language profiler and measurement tools for Phase 2. Prefer function-level,
line-level, allocation-level, and flamegraph evidence over guesses; capture both
runtime and memory hotspots. For mixed-language repositories, profile across
language boundaries.

- Python: `cProfile`, `line_profiler`, `py-spy`, `scalene`, `memory_profiler`, `tracemalloc`, `pytest-benchmark`, `timeit`, `perf_counter`.
- R: `Rprof`, `profvis`, `bench`, `microbenchmark`, `lineprof`, `lobstr`, `pryr`, `data.table` verbose diagnostics.
- C++: `perf`, `gprof`, Callgrind, Valgrind Massif, `heaptrack`, VTune, Instruments, sanitizer allocation traces, compiler vectorization reports.
- C: `perf`, `gprof`, Callgrind, Valgrind Massif, `heaptrack`, VTune, sanitizer traces, compiler vectorization reports.
- Rust: `cargo bench`, Criterion, `cargo flamegraph`, `perf`, `heaptrack`, DHAT, `cargo asm`, `cargo llvm-lines`, allocation counters.
- Go: `go test -bench` + `benchstat`, `pprof` (CPU/heap), `runtime/trace`, escape analysis via `go build -gcflags='-m'`.
- Java: JMH, JFR, async-profiler, GC logs (`-Xlog:gc`), JITWatch, VisualVM.
- Julia: `@time`/`@btime`, `Profile`/`PProf`, allocation tracking (`--track-allocation=user`), `@code_warntype`.
- JavaScript/TypeScript: `node --prof`/`--cpu-prof`, clinic.js, `performance.now()`, browser DevTools performance/memory panels.
- Assembly: `perf`, hardware counters, `llvm-mca`, `objdump`, `uops.info`, VTune, `pmu-tools`, branch-miss and cache-miss counters.
- General: `/usr/bin/time -v`, `hyperfine`, flamegraphs, OS-level memory tools, process-level RSS sampling, benchmark harnesses supplied by the user.
