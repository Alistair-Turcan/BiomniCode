#' BayesSpace: A package for processing spatial transcriptomes
#'
#' Tools for clustering and enhancing the resolution of spatial gene expression
#' experiments. BayesSpace clusters a low-dimensional representation of the gene
#' expression matrix, incorporating a spatial prior to encourage neighboring
#' spots to cluster together. The method can enhance the resolution of the
#' low-dimensional representation into "sub-spots", for which features such as
#' gene expression or cell type composition can be imputed.
#'
#' @details
#' For an overview of the functionality provided by the package, please see the
#' vignette:
#' \code{vignette("BayesSpace", package="BayesSpace")}
#'
#'
#' @name BayesSpace
#'
#' @keywords internal
#'
#' @importFrom Rcpp evalCpp
#' @importFrom SummarizedExperiment assayNames
#' @importFrom SingleCellExperiment altExpNames
#' @useDynLib BayesSpace
#' @useDynLib BayesSpace, .registration=TRUE
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
    ns <- environment(.onLoad)
    delayedAssign("DR_data", DirichletReg::DR_data,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("DirichReg", DirichletReg::DirichReg,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("xgboost", xgboost::xgboost,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("xgb.DMatrix", xgboost::xgb.DMatrix,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("xgb.train", xgboost::xgb.train,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("xgb.params", xgboost::xgb.params,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("logNormCounts", scater::logNormCounts,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("runPCA", scater::runPCA,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("uniquifyFeatureNames", scater::uniquifyFeatureNames,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("getTopHVGs", scran::getTopHVGs,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("modelGeneVar", scran::modelGeneVar,
                  eval.env = baseenv(), assign.env = ns)
    delayedAssign("BiocFileCache", BiocFileCache::BiocFileCache, eval.env = baseenv(), assign.env = ns)
    delayedAssign("bfcrpath", BiocFileCache::bfcrpath, eval.env = baseenv(), assign.env = ns)
    delayedAssign("url.exists", RCurl::url.exists, eval.env = baseenv(), assign.env = ns)
    delayedAssign("read_parquet", arrow::read_parquet, eval.env = baseenv(), assign.env = ns)
    delayedAssign("h5createDataset", rhdf5::h5createDataset, eval.env = baseenv(), assign.env = ns)
    delayedAssign("h5createFile", rhdf5::h5createFile, eval.env = baseenv(), assign.env = ns)
    delayedAssign("h5createGroup", rhdf5::h5createGroup, eval.env = baseenv(), assign.env = ns)
    delayedAssign("h5ls", rhdf5::h5ls, eval.env = baseenv(), assign.env = ns)
    delayedAssign("h5read", rhdf5::h5read, eval.env = baseenv(), assign.env = ns)
    delayedAssign("h5write", rhdf5::h5write, eval.env = baseenv(), assign.env = ns)
}
