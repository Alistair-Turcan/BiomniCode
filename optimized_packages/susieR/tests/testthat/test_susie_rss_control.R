test_that("susie_rss_control returns the documented defaults", {
  expect_identical(
    susie_rss_control(),
    list(
      mismatch_estimator = "mle",
      mixture_reference_p = 5e-8,
      qc_eigen_tol_rel = 1e-3,
      qc_eigen_tol_abs = 0,
      artifact_threshold = 0.1,
      sensitivity_threshold = 30,
      r_tol = 1e-8,
      check_input = FALSE,
      check_prior = TRUE
    )
  )
  expect_false(inherits(susie_rss_control(), "susie_rss_control"))
  expect_identical(eval(formals(summary_stats_constructor)$R_sensitivity_threshold),
                   30)
  expect_identical(eval(formals(ss_mixture_constructor)$R_sensitivity_threshold),
                   30)
  expect_identical(eval(formals(summarize_R_bf_attenuation)$threshold), 30)
})

test_that("susie_rss accepts partial control lists", {
  obj <- suppressWarnings(susie_rss(
    z = c(2, 0), R = diag(2), n = 100, L = 1,
    R_mismatch = "eb_mix",
    control = list(mismatch_estimator = "map",
                   mixture_reference_p = 1e-4),
    init_only = TRUE
  ))

  expect_equal(obj$params$R_mismatch_method, "map")
  expect_equal(obj$params$eb_mix_ref, 1e-4)
  expect_equal(obj$params$artifact_threshold, 0.1)
  expect_equal(obj$params$R_sensitivity_threshold, 30)
})

test_that("susie_rss control rejects malformed input", {
  expect_error(
    susie_rss(z = 0, R = diag(1), control = "mle", init_only = TRUE),
    "control must be a list"
  )
  expect_error(
    susie_rss(z = 0, R = diag(1),
              control = list(mismatch_method = "map"), init_only = TRUE),
    "Unknown control parameter: mismatch_method"
  )
  expect_error(
    susie_rss(z = 0, R = diag(1),
              control = list(1e-8), init_only = TRUE),
    "uniquely named"
  )
})

test_that("legacy RSS control arguments are not top-level arguments", {
  legacy <- c(
    "R_mismatch_method", "eb_mix_ref", "eig_delta_rel", "eig_delta_abs",
    "artifact_threshold", "R_sensitivity_threshold", "r_tol",
    "check_input", "check_prior"
  )
  expect_false(any(legacy %in% names(formals(susie_rss))))
})
