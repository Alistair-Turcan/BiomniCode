//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

double Korg_Binom(double t1, arma::vec & mu, arma::vec & g);
double K1_adj_Binom(double t1, arma::vec & mu, arma::vec & g, double q);
double K2_Binom(double t1, arma::vec & mu, arma::vec & g);
Rcpp::List getroot_K1_Binom(double init, arma::vec & mu, arma::vec & g, double q, double tol, int maxiter = 1000);
Rcpp::List Get_Saddle_Prob_Binom(double zeta,  arma::vec & mu, arma::vec & g, double q, bool logp=false);

// BIOMNI-OPT it22: List-free slow-path cores with shared per-marker
// invariants. Element-wise FP operations are identical to the exported
// originals; only redundant recomputation is removed.
struct KInvBinom {
  arma::vec om;    // 1 - mu
  arma::vec mug;   // mu % g
  arma::vec g2;    // arma::pow(g, 2)
  arma::vec ommu;  // (1 - mu) % mu
};
void computeKInvBinom(arma::vec & mu, arma::vec & g, KInvBinom & inv);
void K1K2_joint_Binom(double t1, arma::vec & mu, arma::vec & g, double q, const KInvBinom & inv, double & k1, double & k2);
void getroot_K1_Binom_core(double init, arma::vec & mu, arma::vec & g, double q, double tol, int maxiter, const KInvBinom & inv, double & root, int & niter, bool & Isconverge);
void Get_Saddle_Prob_Binom_core(double zeta, arma::vec & mu, arma::vec & g, double q, bool logp, const KInvBinom & inv, double & pval, bool & isSaddle);
Rcpp::List SPA_binary(arma::vec & mu, arma::vec & g, double q, double qinv, double pval_noadj, double tol, bool logp = false);
double Korg_fast_Binom(double t1, arma::vec & mu, arma::vec & g,  arma::vec & gNA, arma::vec & gNB, arma::vec & muNA, arma::vec & muNB, double NAmu, double NAsigma);
double K1_adj_fast_Binom(double t1, arma::vec & mu, arma::vec & g, double q, arma::vec & gNA, arma::vec & gNB, arma::vec & muNA, arma::vec & muNB,  double NAmu, double NAsigma);
double K2_fast_Binom(double t1, arma::vec & mu, arma::vec & g, arma::vec & gNA, arma::vec & gNB, arma::vec & muNA, arma::vec & muNB,  double NAmu, double NAsigma);
Rcpp::List getroot_K1_fast_Binom(double init, arma::vec & mu, arma::vec & g, double q, arma::vec & gNA, arma::vec & gNB, arma::vec & muNA, arma::vec & muNB,  double NAmu, double NAsigma, double tol, int maxiter = 1000);
Rcpp::List Get_Saddle_Prob_fast_Binom(double zeta,  arma::vec & mu, arma::vec & g, double q, arma::vec & gNA, arma::vec & gNB, arma::vec & muNA, arma::vec & muNB,  double NAmu, double NAsigma, bool logp=false);
Rcpp::List SPA_binary_fast(arma::vec & mu, arma::vec & g, double q, double qinv, double pval_noadj, bool logp, arma::vec & gNA, arma::vec & gNB, arma::vec & muNA, arma::vec & muNB,  double NAmu, double NAsigma, double tol);
