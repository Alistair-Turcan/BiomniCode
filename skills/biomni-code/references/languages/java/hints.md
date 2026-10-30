# Java optimization hints

Use these hints when optimizing Java implementation code, JVM services, data pipelines, build tools, and JVM-hosted scientific libraries. Apply only when profiling indicates the code path matters and behavior is preserved.

1. Optimize by measuring with JMH for microbenchmarks and JFR or async-profiler for whole-program evidence; account for JIT warm-up before trusting any number.
2. Optimize by reducing allocation rate in hot paths; young-generation GC pressure is the most common JVM bottleneck.
3. Optimize by reusing buffers, builders, and scratch objects instead of allocating per call when behavior allows it.
4. Optimize by using `StringBuilder` (or `String.join`) instead of repeated `+` concatenation in loops.
5. Optimize by avoiding boxed primitives (`Integer`, `Double`) in hot collections; use primitive-specialized libraries such as fastutil, Eclipse Collections, or HPPC.
6. Optimize by preferring arrays or `ArrayList` over `LinkedList`; pointer-chasing layouts destroy cache locality.
7. Optimize by preallocating `ArrayList`, `HashMap`, and `StringBuilder` with expected capacity to avoid repeated growth and rehashing.
8. Optimize by choosing the right map: `HashMap` for general use, `EnumMap` for enum keys, sorted arrays plus binary search for read-mostly lookup tables.
9. Optimize by avoiding Java Streams in proven hot loops when an indexed loop allocates less and inlines better; streams are fine elsewhere.
10. Optimize by hoisting invariant computation, regex compilation (`Pattern.compile` once, statically), and formatter construction out of loops.
11. Optimize by avoiding reflection and `Method.invoke` in hot paths; cache `MethodHandle`s or use code generation when reflection is unavoidable.
12. Optimize by avoiding unnecessary synchronization; prefer `ConcurrentHashMap`, atomics, or thread-local state over coarse `synchronized` blocks.
13. Optimize by reducing lock hold times and sharding contended structures instead of adding more threads to a contended lock.
14. Optimize by using `BufferedReader`/`BufferedWriter`/`BufferedInputStream` for all file and socket IO; unbuffered IO is syscall-bound.
15. Optimize by reading files in bulk (`Files.readAllBytes`, `FileChannel`, memory-mapped buffers) when the workload is IO-dominated and sizes permit.
16. Optimize by choosing faster serialization (Kryo, protobuf, FlatBuffers, Jackson `ObjectMapper` reuse with `ObjectReader`/`ObjectWriter`) when (de)serialization dominates and formats are contractual.
17. Optimize by reusing `ObjectMapper`, `Gson`, and similar heavy factories; constructing them per call is a classic hidden cost.
18. Optimize by avoiding exceptions for control flow; exception construction captures a stack trace and is expensive.
19. Optimize by using lazy logging (`log.debug("{}", value)` or supplier-based guards) so message construction is skipped when the level is disabled.
20. Optimize by enabling the G1 or ZGC/Shenandoah collector and sizing the heap deliberately (`-Xms` = `-Xmx` for services) instead of accepting defaults.
21. Optimize by checking for megamorphic call sites and excessive interface dispatch with JFR; monomorphic sites inline, megamorphic ones do not.
22. Optimize by keeping hot methods small enough to inline (default inline budget favors small methods) and avoiding deep call chains in inner loops.
23. Optimize by using `final` fields and immutable value objects where practical; they enable JIT optimizations and safe sharing without copies.
24. Optimize by batching database and network calls; per-row round trips dominate almost every other JVM-level concern.
25. Optimize by using primitive streams (`IntStream`, `LongStream`, `DoubleStream`) or plain loops to avoid boxing in numeric pipelines.
26. Optimize by avoiding `Vector`, `Hashtable`, and other legacy synchronized collections; use unsynchronized collections with external coordination.
27. Optimize by parallelizing independent work with `ForkJoinPool`/parallel streams or structured concurrency, sized to cores, only when tasks are coarse enough to amortize scheduling.
28. Optimize by watching for false sharing and padding contended counters (`@Contended` where available) in highly concurrent code.
29. Optimize by considering ahead-of-time or class-data-sharing options (AppCDS, GraalVM native image) for startup-dominated CLI tools when deployment constraints allow.
30. Optimize by reading GC logs and allocation profiles first; attack the top allocation sites and the longest GC pauses before micro-tuning code.
