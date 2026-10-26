#include <cpp11.hpp>
#include <cpp11armadillo.hpp>
#include "mr_ash.h"

using namespace cpp11;
using namespace arma;

// Random permutation index vector of length p * numiter (0-based).
[[cpp11::register]]
integers random_order(int p, int numiter) {
  return as_integers(random_order_impl(p, numiter));
}

// Mr.ASH coordinate-ascent in sufficient-statistic form.
[[cpp11::register]]
writable::list caisa_cpp(const doubles_matrix<>& X,
                         const doubles& w, const doubles& sa2,
                         const doubles& pi_init, const doubles& beta_init,
                         const doubles& r_init, double sigma2,
                         const integers& o_r,
                         int maxiter, int miniter,
                         double convtol, double epstol, std::string method_q,
                         bool updatepi, bool updatesigma,
                         bool verbose) {

  // cpp11 inputs -> Armadillo. pi, beta, r are mutated -> own their memory.
  const mat X_mat   = as_Mat(X);
  const vec w_vec   = as_Col(w);
  const vec sa2_vec = as_Col(sa2);
  const uvec o      = as_uvec(o_r);
  vec pi    = conv_to<vec>::from(as_Col(pi_init));
  vec beta  = conv_to<vec>::from(as_Col(beta_init));
  vec r     = conv_to<vec>::from(as_Col(r_init));

  const int n = X_mat.n_rows;
  const int p = X_mat.n_cols;
  const int K = sa2_vec.n_elem;

  // Per-iter per-coordinate prior weights (mixture precision + X*X'/sigma2).
  mat S2inv = 1.0 / outerAddition(1.0 / sa2_vec, w_vec);
  S2inv.row(0).fill(epstol);

  vec varobj(maxiter);
  vec piold, betaold;
  int iter = 0;
  int i = 0;

  for (iter = 0; iter < maxiter; iter++) {
    double a1 = 0.0, a2 = 0.0;
    piold   = pi;
    betaold = beta;
    pi.fill(0);

    // Coordinate-ascent sweep (random order given by o)
    for (int j = 0; j < p; j++) {
      updatebetaj(X_mat.col(o(i)), w_vec(o(i)), beta(o(i)), r, piold, pi,
                  sigma2, sa2_vec, S2inv.col(o(i)), a1, a2, o(i), p, epstol);
      i++;
    }

    // Variational objective (first term)
    varobj(iter) = dot(r, r) - dot(square(beta), w_vec) + a1;

    // Optionally update sigma2
    if (updatesigma) {
      if (method_q == "sigma_indep_q") {
        sigma2 = (varobj(iter) + p * (1.0 - pi(0)) * sigma2)
                 / (n + p * (1.0 - pi(0)));
      } else if (method_q == "sigma_dep_q") {
        sigma2 = varobj(iter) / n;
      }
    }

    // Freeze piold for objective computation when updating pi
    if (updatepi) piold = pi;

    // Variational objective (full expression)
    varobj(iter) = varobj(iter) / sigma2 / 2.0
                   + log(2.0 * M_PI * sigma2) / 2.0 * n
                   - dot(pi, log(piold + epstol)) * p + a2;
    for (int j = 1; j < K; j++) {
      varobj(iter) += pi(j) * log(sa2_vec(j)) * p / 2.0;
    }

    // Restore pi if we are not updating it
    if (!updatepi) pi = piold;

    // Convergence: beta change small, or objective non-decreasing
    if (iter >= miniter - 1) {
      double beta_norm = norm(beta, 2);
      if (norm(betaold - beta, 2) < convtol * std::max(1.0, beta_norm)) {
        iter++;
        break;
      }
      if (iter > 0 && varobj(iter) > varobj(iter - 1)) break;
    }
  }

  if (verbose) {
    Rprintf("Mr.ASH terminated at iteration %d: max|beta|=%.4e, sigma2=%.4e, pi0=%.4f\n",
            iter, max(abs(beta)), sigma2, pi(0));
  }

  using namespace cpp11::literals;
  return writable::list({
    "beta"_nm   = as_doubles(beta),
    "sigma2"_nm = as_sexp(sigma2),
    "pi"_nm     = as_doubles(pi),
    "iter"_nm   = as_sexp(iter),
    "varobj"_nm = as_doubles(vec(varobj.subvec(0, iter - 1)))
  });
}
