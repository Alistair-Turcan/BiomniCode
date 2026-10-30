# Python optimization hints

Use these hints when optimizing Python, NumPy, pandas, SciPy, PyTorch, JAX, Numba, Cython, pybind11, or Python-orchestrated workflows. Apply only when profiling indicates the code path matters and behavior is preserved.

1. Optimize by removing recursive functions when an iterative version avoids stack overhead and preserves behavior.
2. Optimize by removing redundant calculations inside loops.
3. Optimize by vectorizing operations with NumPy, pandas, SciPy, PyTorch, JAX, or other array libraries when appropriate.
4. Optimize by using NumPy arrays for numeric loops while returning values as the original data type.
5. Optimize by using list comprehensions instead of repeated append loops for simple transformations.
6. Optimize by using generator expressions to avoid materializing large intermediate lists.
7. Optimize by using efficient data structures such as `set`, `dict`, `deque`, `heapq`, `array`, `Counter`, `defaultdict`, or `bisect`.
8. Optimize by using efficient built-in functions such as `sum`, `min`, `max`, `sorted`, `any`, `all`, `zip`, `map`, and `enumerate`.
9. Optimize by storing temporary variables to avoid repeated attribute lookups or repeated computations.
10. Optimize by reducing repeated calls to global variables inside hot loops.
11. Optimize by removing unnecessary operations, casts, branches, wrappers, and conversions.
12. Optimize by directly computing expressions instead of using slow Python-level loops.
13. Optimize by avoiding repeated string concatenation and using `join`, `StringIO`, or buffered writers.
14. Optimize by avoiding pandas `iterrows` and replacing it with vectorized operations, `itertuples`, or array-level code.
15. Optimize by replacing pandas `apply` with vectorized column operations when possible.
16. Optimize by avoiding unnecessary `DataFrame`, `Series`, list, dict, and ndarray copies.
17. Optimize by using categorical dtypes for repeated string labels in pandas.
18. Optimize by avoiding object dtype arrays when numeric, categorical, or fixed-width dtypes can be used.
19. Optimize by preallocating arrays instead of repeatedly growing NumPy arrays or large Python containers.
20. Optimize by avoiding repeated `np.concatenate`, `np.append`, or DataFrame concatenation inside loops.
21. Optimize by using broadcasting instead of explicit nested loops.
22. Optimize by using `scipy.sparse` or other sparse formats instead of dense arrays for mostly zero data.
23. Optimize by caching expensive pure function results with memoization when inputs repeat.
24. Optimize by moving invariant computations outside loops.
25. Optimize by batching small operations to reduce interpreter overhead.
26. Optimize by binding frequently used functions or attributes to local variables inside hot loops.
27. Optimize by using multiprocessing, `concurrent.futures`, joblib, Dask, Ray, or native library threading for independent work.
28. Optimize by using Numba, Cython, pybind11, Rust extensions, or compiled kernels for unavoidable tight loops.
29. Optimize by avoiding unnecessary file reads, writes, flushes, serialization, and deserialization inside loops.
30. Optimize by profiling before and after each change with appropriate Python runtime and memory profilers.
