# iter13: bitwise-identical fast paths for hot R-level operations in
# process.locus / read.ld / decompose.ld. Each is verified against its base-R
# equivalent by bench/test_iter13_bitwise.R; see comments for the exact
# equivalence argument.

# x %in% table for character vectors, via interned-CHARSXP pointer membership
# in C++. The C++ returns NULL if any string has a non-ASCII byte (where base
# match() may compare across encodings); fall back to base %in% then.
in_set = function(x, table) {
  r = in_set_cpp(x, table)
  if (is.null(r)) return(x %in% table)
  r
}

# svd(X, nu = 0): performs the identical La.svd(X, 0, min(n,p)) call that
# base::svd makes (La.svd repeats the same input checks with the same error
# messages, so skipping svd()'s redundant as.matrix/is.finite pass is
# behavior-preserving), and replaces t(La.res$vt) with a C++ transpose (a pure
# permutation of doubles - no floating-point operations, hence bitwise-equal).
svd_v_only = function(X) {
  if (!is.matrix(X) || !is.numeric(X) || is.complex(X)) return(svd(X, nu = 0))
  La.res = La.svd(X, 0, min(nrow(X), ncol(X)))
  list(d = La.res$d, v = transpose_cpp(La.res$vt))
}

# which(x %in% table) fused (integer indices; base fallback on non-ASCII)
which_in_set = function(x, table) {
  r = which_in_set_cpp(x, table)
  if (is.null(r)) return(which(x %in% table))
  r
}

# df[idx, ] for a plain data.frame, replicated bitwise: same column subsets and
# names; row.names materialized exactly as [.data.frame does (compact
# c(NA,-n) row.names become the selected 1-based indices; materialized
# row.names are subsetted). Falls back to base [ for non-plain inputs (extra
# classes or column attributes).
df_rows = function(df, idx) {
  if (!identical(class(df), "data.frame") || any(vapply(df, function(col) !is.null(attributes(col)), logical(1)))) return(df[idx, ])
  cols = lapply(df, function(col) col[idx])
  rn = attr(df, "row.names")
  rn = if (length(rn) == 2L && is.na(rn[1])) idx else rn[idx]
  structure(cols, class = "data.frame", row.names = rn, names = names(df))
}

# iter16: ASCII-only tolower/toupper (C++). Bitwise-identical to base for
# pure-ASCII strings (ASCII case mapping is locale-independent); the C++
# returns NULL if any element has a non-ASCII byte and we fall back to base.
ascii_tolower = function(x) {
  r = ascii_tolower_cpp(x)
  if (is.null(r)) return(tolower(x))
  r
}
ascii_toupper = function(x) {
  r = ascii_toupper_cpp(x)
  if (is.null(r)) return(toupper(x))
  r
}

