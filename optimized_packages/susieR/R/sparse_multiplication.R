# @title Computes standardized.X %*% b using sparse multiplication trick
# @param X an n by p unstandardized matrix with three attributes:
# attr(X,"scaled:center"), attr(X,"scaled:scale") and attr(X,"d")
# @param b a p vector
# @return an n vector
#
#' @importFrom Matrix t
#' @importFrom Matrix tcrossprod
compute_Xb <- function(X, b, cm = NULL, csd = NULL) {
  # Scaling statistics: explicit arguments (data$X_cm/data$X_csd) first,
  # attribute fallback for hand-built or attribute-carrying X (test
  # fixtures, trend-filtering matrices, downstream callers).
  if (is.null(cm)) cm <- attr(X, "scaled:center")
  if (is.null(csd)) csd <- attr(X, "scaled:scale")

  # Scale Xb.
  if (!is.null(attr(X, "matrix.type"))) {

    # When X is a trend filtering matrix.
    scaled.Xb <- compute_tf_Xb(attr(X, "order"), b / csd)
  } else {

    # When X is an ordinary sparse/dense matrix.
    #
    # Zero-effect fast path: for an all-zero b (the all-zero
    # initialization, and null effects whose posterior mean is exactly
    # +/-0), the dense matvec below is bitwise a +0 vector: every dot
    # product accumulates +/-0 terms from a +0 seed and (+0) + (+/-0) = +0,
    # and sum(cm * 0 / csd) = +0 when cm is finite and csd is finite and
    # nonzero, so scaled.Xb - sum(...) = +0 - +0 = +0 (verified empirically
    # against tcrossprod on the benchmark data, including mixed signed-zero
    # b). Skipping the O(np) sweep in that case is bitwise-identical.
    # Non-finite cm/csd (possible only with non-finite columns in X) keep
    # the original path to preserve NaN/Inf propagation.
    if (is.matrix(X) && is.double(X) && is.double(b) &&
        isTRUE(all(b == 0)) &&
        all(is.finite(cm)) && all(is.finite(csd)) && all(csd != 0))
      return(rep(0, nrow(X)))
    scaled.Xb <- tcrossprod(X, t(b / csd))
    # Fused centering for the plain dense case: bitwise-identical to the
    # shared tail below (same per-element divides in double, long-double
    # sequential accumulation matching base sum(), same scalar
    # subtraction). Sparse/classed tcrossprod results fall through.
    if (is.matrix(scaled.Xb) && !is.object(scaled.Xb) &&
        is.double(scaled.Xb) &&
        length(cm) == length(b) && length(csd) == length(b)) {
      fused <- xb_post_cpp(scaled.Xb, cm, b, csd)
      if (length(fused) == nrow(X))
        return(fused)
    }
  }

  # Center Xb.
  Xb <- scaled.Xb - sum(cm * b / csd)
  return(as.numeric(Xb))
}

# @title Computes t(standardized.X) %*% y using sparse multiplication trick
# @param X an n by p unstandardized matrix with three attributes:
# attr(X,"scaled:center"), attr(X,"scaled:scale") and attr(X,"d")
# @param y an n vector
# @return a p vector
#
#' @importFrom Matrix t
#' @importFrom Matrix crossprod
compute_Xty <- function(X, y, cm = NULL, csd = NULL) {
  if (is.null(cm)) cm <- attr(X, "scaled:center")
  if (is.null(csd)) csd <- attr(X, "scaled:scale")
  ytX <- crossprod(y, X)

  # Scale Xty.
  if (!is.null(attr(X, "matrix.type"))) {

    # When X is a trend filtering matrix.
    scaled.Xty <- compute_tf_Xty(attr(X, "order"), y) / csd
  } else if (is.matrix(X) && is.double(X) && is.double(ytX) &&
             is.double(cm) && is.double(csd) &&
             length(cm) == length(csd)) {

    # When X is an ordinary dense double matrix: fused post-matvec
    # arithmetic (bitwise-identical to t(ytX/csd) - cm/csd*sum(y); t() is
    # an exact permutation and the kernel keeps R's per-element op order).
    # Falls through to the R chain on any shape mismatch.
    fused <- xty_post_cpp(ytX, cm, csd, sum(y))
    if (length(fused) == length(csd))
      return(fused)
    scaled.Xty <- t(ytX / csd)
  } else {

    # When X is an ordinary sparse/dense matrix.
    scaled.Xty <- t(ytX / csd)
  }

  # Center Xty.
  centered.scaled.Xty <- scaled.Xty - cm / csd * sum(y)
  return(as.numeric(centered.scaled.Xty))
}

# @title Computes t(standardized.X) %*% standardized.X using attributes
# @param X an n by p unstandardized matrix with three attributes:
# attr(X,"scaled:center"), attr(X,"scaled:scale") and attr(X,"d")
# @return a p by p matrix representing (scaled X)'(scaled X)
#
#' @importFrom Matrix crossprod
compute_XtX <- function(X, cm = NULL, csd = NULL) {
  if (is.null(cm)) cm <- attr(X, "scaled:center")
  if (is.null(csd)) csd <- attr(X, "scaled:scale")
  n <- nrow(X)
  colsums_X <- colSums(X)

  if (!is.null(attr(X, "matrix.type"))) {
    stop("compute_XtX not yet implemented for trend filtering matrices")
  }

  # Compute raw X'X
  XtX_raw <- crossprod(X)

  # Scale columns and rows by 1/csd
  XtX_scaled <- sweep(sweep(XtX_raw, 1, csd, "/"), 2, csd, "/")

  # Adjust for centering
  XtX_centered_scaled <- XtX_scaled -
    n * outer(cm/csd, cm/csd) -
    outer(cm/csd, (colsums_X - n*cm)/csd) -
    outer((colsums_X - n*cm)/csd, cm/csd)

  return(XtX_centered_scaled)
}

# @title Computes M %* %t(standardized.X) using sparse multiplication trick
# @param M a L by p matrix
# @param X an n by p unstandardized matrix with three attributes:
# attr(X,"scaled:center"), attr(X,"scaled:scale") and attr(X,"d")
# @return a L by n matrix
#
#' @importFrom Matrix t
compute_MXt <- function(M, X, cm = NULL, csd = NULL) {
  if (is.null(cm)) cm <- attr(X, "scaled:center")
  if (is.null(csd)) csd <- attr(X, "scaled:scale")

  if (!is.null(attr(X, "matrix.type"))) {

    # When X is a trend filtering matrix.
    return(as.matrix(t(apply(M, 1, function(b) compute_Xb(X, b)))))
  } else {

    # When X is an ordinary sparse/dense matrix.
    #
    # Plain dense fast path: fused pre/post arithmetic around the *same*
    # BLAS gemm and gemv calls as the original expression. mxt_pre_cpp is
    # bitwise t(M)/csd (t() is an exact permutation; the divide recycles
    # csd down columns) and mxt_post_cpp is bitwise t(tmp) - v (exact
    # permutation; v recycles down columns), so the gemm/gemv receive
    # bitwise-identical inputs and the result is bitwise-identical.
    # Classed/sparse inputs and any shape mismatch fall through to the
    # original expression.
    if (!is.object(X) && is.matrix(X) && is.double(X) &&
        !is.object(M) && is.matrix(M) && is.double(M) &&
        is.double(cm) && is.double(csd) &&
        ncol(M) == ncol(X) && length(cm) == ncol(X) &&
        length(csd) == ncol(X)) {
      pre <- mxt_pre_cpp(M, csd, nrow(M), ncol(M))
      if (length(pre) == nrow(M) * ncol(M)) {
        dim(pre) <- c(ncol(M), nrow(M))
        tmp  <- X %*% pre
        v    <- drop(M %*% (cm / csd))
        post <- mxt_post_cpp(tmp, v, nrow(M), nrow(X))
        if (length(post) == nrow(M) * nrow(X)) {
          dim(post) <- c(nrow(M), nrow(X))
          return(post)
        }
      }
    }
    return(as.matrix(t(X %*% (t(M) / csd)) - drop(M %*% (cm / csd))))
  }

  # This should be the same as
  #
  #   t(apply(M, 1, function(b) compute_Xb(X, b))))
  #
  # as well as
  #
  #   M %*% (t(X)/csd) - drop(tcrossprod(M,t(cm/csd)))
  #
  # but should be more memory-efficient.
}
