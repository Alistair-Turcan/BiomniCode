# R optimization hints

Use these hints when optimizing R, data.table, tidyverse, Matrix, Rcpp, cpp11, Bioconductor, or R package code. Apply only when profiling indicates the code path matters and behavior is preserved.

1. Optimize by avoiding growing vectors, lists, matrices, or data frames inside loops.
2. Optimize by preallocating vectors, matrices, lists, and data frames before filling them.
3. Optimize by vectorizing operations instead of using explicit loops where vectorization preserves behavior.
4. Optimize by using matrix operations instead of nested R loops.
5. Optimize by using `data.table` instead of repeated base `data.frame` operations on large tables.
6. Optimize by using dplyr carefully and avoiding unnecessary intermediate tibbles or grouped recomputation.
7. Optimize by avoiding repeated `rbind` or `cbind` inside loops.
8. Optimize by accumulating results in a list and combining once with `data.table::rbindlist` or `do.call`.
9. Optimize by using `vapply` instead of `sapply` when output type is known.
10. Optimize by using `lapply`, `vapply`, `Map`, or vectorized functions instead of slow manual list construction when appropriate.
11. Optimize by avoiding `apply` on data frames when it causes costly coercion.
12. Optimize by using `rowSums`, `colSums`, `rowMeans`, `colMeans`, `pmin`, `pmax`, and `tabulate` instead of slower generic operations.
13. Optimize by using `matrixStats` for fast row-wise and column-wise statistics.
14. Optimize by avoiding unnecessary copies of large R objects.
15. Optimize by modifying `data.table` objects by reference with `:=`, `set`, `setnames`, `setkey`, and related functions.
16. Optimize by avoiding repeated factor-to-character or character-to-factor conversions.
17. Optimize by using factors, integer encodings, or dictionary mappings for repeated string labels.
18. Optimize by avoiding unnecessary `as.data.frame`, `as.matrix`, `as_tibble`, and type conversions.
19. Optimize by moving invariant calculations outside loops.
20. Optimize by caching repeated calculations, parsed objects, model matrices, formulas, and design matrices.
21. Optimize by using `Matrix` sparse matrices for mostly zero data.
22. Optimize by avoiding dense conversion of sparse single-cell, spatial, or genetics matrices.
23. Optimize by using `fastmatch`, `data.table` joins, keyed tables, named vectors, or environments instead of repeated slow `match` operations.
24. Optimize by using keyed `data.table` joins instead of repeated `merge` calls.
25. Optimize by avoiding repeated expensive function calls inside `mutate`, `summarize`, `lapply`, or grouped operations.
26. Optimize by moving unavoidable tight loops into Rcpp, cpp11, C, C++, Fortran, or optimized package backends.
27. Optimize by using `parallel`, `future`, `foreach`, BiocParallel, `data.table` threading, or native BLAS threading for independent work.
28. Optimize by avoiding unnecessary printing, plotting, progress updates, and console output inside hot loops.
29. Optimize by reading large files with `fread`, `vroom`, `arrow`, `readr`, `fst`, `qs`, or other fast IO tools instead of slow base readers.
30. Optimize by profiling before and after each change with `Rprof`, `profvis`, `bench`, `microbenchmark`, memory tools, or allocation tracing.
