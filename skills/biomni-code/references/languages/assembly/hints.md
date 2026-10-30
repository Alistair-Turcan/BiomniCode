# Assembly optimization hints

Use these hints when optimizing handwritten assembly, compiler-generated assembly, SIMD kernels, ABI boundaries, or microarchitectural hotspots. Apply only when profiling indicates the code path matters and behavior is preserved.

1. Optimize by first confirming the hotspot with hardware counters rather than rewriting non-hot assembly.
2. Optimize by preserving ABI requirements exactly, including calling convention, callee-saved registers, stack alignment, red-zone rules, and vector register preservation rules.
3. Optimize by minimizing memory traffic before minimizing instruction count.
4. Optimize by keeping hot data in registers and avoiding unnecessary spills and reloads.
5. Optimize by reducing load-use latency through instruction scheduling.
6. Optimize by unrolling loops only when it improves throughput without excessive code size or instruction-cache pressure.
7. Optimize by aligning hot loops and branch targets when profiling or architecture guidance shows a benefit.
8. Optimize by avoiding partial-register stalls and false dependencies on architectures where they matter.
9. Optimize by replacing unpredictable branches with conditional moves, masks, predication, or table lookups when beneficial.
10. Optimize by keeping predictable branches when they are cheaper than branchless alternatives.
11. Optimize by using SIMD instructions for data-parallel work when alignment, data layout, and tail handling are correct.
12. Optimize by selecting the right SIMD width for the target CPU instead of assuming the widest vector is fastest.
13. Optimize by providing scalar fallbacks or dispatch paths when using architecture-specific instructions.
14. Optimize by using non-temporal loads or stores only for streaming data that will not be reused soon.
15. Optimize by improving data alignment for vector loads and stores where misalignment is costly.
16. Optimize by avoiding unnecessary gathers and scatters unless the access pattern justifies them.
17. Optimize by transforming data layout to reduce shuffle, permute, gather, or scatter overhead when behavior permits.
18. Optimize by reducing dependency chains and increasing instruction-level parallelism.
19. Optimize by balancing port pressure across execution units on the target microarchitecture.
20. Optimize by using fused multiply-add or other fused instructions only when numerical behavior remains acceptable.
21. Optimize by using integer strength reduction for expensive operations such as division or modulo when exact semantics are preserved.
22. Optimize by replacing repeated address recomputation with efficient addressing modes or maintained pointers.
23. Optimize by using `lea` and scaled addressing effectively without creating excessive dependency chains.
24. Optimize by avoiding slow instructions in hot loops when equivalent faster sequences preserve flags and results.
25. Optimize by separating rare paths, error paths, and remainder handling from the common fast loop.
26. Optimize by minimizing function-call overhead in tiny hot kernels through inlining or leaf-function design when ABI constraints allow.
27. Optimize by reducing branch target count and indirect branches in tight dispatch-heavy loops.
28. Optimize by validating performance with realistic input distributions, because microbenchmarks can favor the wrong branch or cache behavior.
29. Optimize by analyzing generated code with `objdump`, `llvm-mca`, `perf`, VTune, hardware counters, and architecture manuals before and after changes.
30. Optimize by documenting assumptions about CPU features, alignment, aliasing, flags, registers, memory ordering, and numerical behavior next to the assembly.
