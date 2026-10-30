# Go optimization hints

Use these hints when optimizing Go implementation code, CLI tools, services, parsers, data pipelines, and Go-backed libraries. Apply only when profiling indicates the code path matters and behavior is preserved.

1. Optimize by measuring with `go test -bench`, `pprof`, and `runtime/trace` before and after each change; never guess at hot paths.
2. Optimize by preallocating slices with `make([]T, 0, capacity)` when the final size is known or estimable.
3. Optimize by preallocating maps with `make(map[K]V, capacity)` to avoid repeated rehashing and growth.
4. Optimize by reusing buffers and scratch slices instead of allocating inside hot loops.
5. Optimize by using `sync.Pool` for frequently allocated short-lived objects when allocation dominates and correctness is preserved.
6. Optimize by using `strings.Builder` or `bytes.Buffer` instead of repeated string concatenation.
7. Optimize by avoiding string-to-`[]byte` conversions in hot paths; each conversion copies.
8. Optimize by avoiding unnecessary `fmt.Sprintf`/`fmt.Println` in hot paths; formatting is reflection-heavy and allocates.
9. Optimize by using `strconv` functions instead of `fmt` for numeric conversions.
10. Optimize by keeping values on the stack; check escape analysis with `go build -gcflags='-m'` and reduce needless escapes to the heap.
11. Optimize by avoiding defer in tight loops; defer has fixed overhead, so hoist it or restructure when profiling shows it matters.
12. Optimize by preferring slices over maps for dense integer-keyed data; arrays and slices have far better cache behavior.
13. Optimize by iterating slices with index or range by value for small element types; range copies each element, so use indices for large structs.
14. Optimize by passing pointers to large structs instead of copying them, while keeping small structs by value.
15. Optimize by avoiding interface dispatch and reflection (`reflect` package) in proven hot paths.
16. Optimize by replacing `interface{}`/`any` with concrete types or generics where the hot path allows it.
17. Optimize by batching work per goroutine; goroutines are cheap but not free, and millions of tiny tasks lose to a bounded worker pool.
18. Optimize by avoiding unbuffered channel ping-pong in tight loops; batch messages or use buffered channels sized to the pipeline.
19. Optimize by reducing lock contention with sharding, `sync.Map` only where it fits, atomics for simple counters, or copy-on-write snapshots.
20. Optimize by avoiding `time.Sleep`-based polling loops when condition variables, channels, or `sync.Cond` express the same behavior.
21. Optimize by using buffered IO (`bufio.Reader`/`bufio.Writer`) for file and network IO; unbuffered syscalls dominate otherwise.
22. Optimize by reading files in large chunks or with `io.Copy`/`io.CopyBuffer` instead of line-at-a-time or byte-at-a-time loops.
23. Optimize by choosing `encoding/gob`, `encoding/json` alternatives, or code-generated codecs (e.g. protobuf, msgpack, easyjson) when serialization dominates and formats are contractual.
24. Optimize by compiling regexes once at package level with `regexp.MustCompile` instead of per-call compilation.
25. Optimize by using `bytes` package functions on `[]byte` instead of converting to string for search and compare.
26. Optimize by sorting with `sort.Slice`/`slices.Sort` on pre-collected data instead of repeated insertion into ordered structures.
27. Optimize by enabling bounds-check elimination: use local variables for repeated slice indexing, iterate with range, and avoid re-slicing patterns the compiler cannot prove safe.
28. Optimize by setting `GOMAXPROCS` deliberately for CPU-bound work and by matching worker counts to available cores.
29. Optimize by building with `-ldflags="-s -w"` for smaller binaries and considering `GOAMD64`/`GOARM64` architecture levels for vectorized stdlib paths when deployment targets allow.
30. Optimize by checking allocation profiles (`go test -bench . -benchmem`, `pprof` heap profiles) and attacking the top allocation sites first.
