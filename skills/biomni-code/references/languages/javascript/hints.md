# JavaScript/TypeScript optimization hints

Use these hints when optimizing JavaScript or TypeScript implementation code, Node.js services and CLIs, browser code, bundlers, parsers, and data pipelines. Apply only when profiling indicates the code path matters and behavior is preserved.

1. Optimize by measuring first: `node --prof`/`--cpu-prof`, `console.time`, `performance.now()`, clinic.js, or browser DevTools performance and memory panels.
2. Optimize by keeping object shapes stable: initialize all properties in the constructor or literal in a consistent order so V8 keeps hidden classes monomorphic.
3. Optimize by avoiding `delete` on object properties; it transitions the object to dictionary mode. Set to `undefined` or use a `Map` instead.
4. Optimize by keeping functions monomorphic: pass values of one consistent type/shape per call site; polymorphic call sites block inlining.
5. Optimize by avoiding `arguments`, `with`, and megamorphic property access in hot functions; they defeat V8 optimization tiers.
6. Optimize by using typed arrays (`Float64Array`, `Int32Array`, `Uint8Array`) for numeric data instead of regular arrays of numbers.
7. Optimize by preallocating arrays with known length and filling by index instead of repeated `push` when the size is known.
8. Optimize by avoiding array methods that allocate (`map`, `filter`, `concat`, spread) in proven hot loops when a single fused loop preserves behavior.
9. Optimize by avoiding sparse arrays and mixed-element-kind arrays; holes and mixed types force slower element kinds.
10. Optimize by using `Map`/`Set` for frequent insertion, deletion, and lookup instead of plain objects with dynamic keys.
11. Optimize by hoisting invariant computation, regex literals/compilation, and function lookups out of loops.
12. Optimize by avoiding repeated property lookups (`a.b.c.d`) in hot loops; bind the leaf to a local variable.
13. Optimize by replacing string concatenation in loops with array-join or template accumulation, and by avoiding repeated `JSON.parse`/`JSON.stringify` of the same values.
14. Optimize by streaming large payloads (`fs.createReadStream`, `stream/pipelines`, web Streams) instead of buffering entire files or responses in memory.
15. Optimize by using `Buffer` operations and `Buffer.concat` for binary data in Node instead of string-based byte handling.
16. Optimize by batching event-loop work: thousands of tiny `setImmediate`/`Promise` continuations cost more than one chunked loop with periodic yields.
17. Optimize by moving CPU-bound work off the event loop with `worker_threads` (Node) or Web Workers (browser) when tasks are coarse enough to amortize transfer costs.
18. Optimize by avoiding synchronous Node APIs (`readFileSync`, `execSync`) on server hot paths; they block every concurrent request.
19. Optimize by caching compiled templates, parsed schemas, and validator functions (e.g. Ajv-compiled validators) instead of rebuilding per call.
20. Optimize by choosing faster serialization where formats are contractual: `JSON.parse` of large payloads is often the bottleneck; consider streaming JSON parsers or binary formats.
21. Optimize by reducing closure allocation inside hot loops; hoist closures that capture loop-invariant state.
22. Optimize by avoiding unnecessary async/await in tight loops; each `await` schedules a microtask, so batch with `Promise.all` or restructure.
23. Optimize by debouncing or batching DOM writes and layout reads in browser code; interleaved read/write forces repeated layout.
24. Optimize by using `requestAnimationFrame` for visual updates and `requestIdleCallback` for deferrable work in browser code.
25. Optimize by reducing bundle and parse cost for startup-dominated tools: lazy-load heavy modules with dynamic `import()` and trim dependency weight.
26. Optimize by avoiding prototype-chain lookups and proxy traps in hot paths; both are far slower than own-property access.
27. Optimize by using `for`/`for-of` loops in proven hot paths; `forEach` with a closure is convenient but allocates and resists inlining.
28. Optimize by checking for accidental quadratic patterns: `array.includes` inside a loop, repeated `indexOf` scans, and string `replace` in loops.
29. Optimize by pinning object property order and avoiding integer-like string keys on objects; both change V8's internal representation.
30. Optimize by reading the memory panel for detached DOM nodes, unbounded caches, and listener leaks; in long-lived JS processes, leaks dominate raw speed issues.
