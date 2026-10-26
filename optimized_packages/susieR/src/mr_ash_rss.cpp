#include <cpp11.hpp>
#include <cpp11armadillo.hpp>
#include "mr_ash_rss.h"

using namespace cpp11;
using namespace arma;
using namespace std;

[[cpp11::register]]
writable::list mr_ash_rss_cpp(const doubles& bhat, const doubles& shat, const doubles& z,
                              const doubles_matrix<>& R, double var_y, int n, double sigma2_e,
                              const doubles& s0, const doubles& w0, const doubles& mu1_init,
                              double tol = 1e-8, int max_iter = 1e5,
                              bool update_w0 = true, bool update_sigma = true,
                              bool compute_ELBO = true, bool standardize = false) {

	// Convert input types
	vec bhat_vec = as_Col(bhat);
	vec shat_vec = as_Col(shat);
	vec z_vec = as_Col(z);
	mat R_mat = as_Mat(R);
	vec s0_vec = as_Col(s0);
	vec w0_vec = as_Col(w0);
	vec mu1_init_vec = as_Col(mu1_init);

	// Call the C++ function
	unordered_map<string, mat> result = mr_ash_rss(bhat_vec, shat_vec, z_vec, R_mat, var_y, n, sigma2_e, s0_vec, w0_vec,
	                                               mu1_init_vec, tol, max_iter, update_w0, update_sigma, compute_ELBO,
	                                               standardize);

	// Convert the result to a named list (matrices returned as doubles_matrix).
	// The unordered_map iteration does not preserve insertion order.
	writable::list ret;
	for (const auto& item : result) {
		cpp11::named_arg na(item.first.c_str());
		na = as_doubles_matrix(item.second);
		ret.push_back(na);
	}

	return ret;
}
