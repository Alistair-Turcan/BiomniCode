
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "UTIL.hpp"
#include <sys/time.h>

arma::vec nb(unsigned int n){
  return(Rcpp::rbinom(n,1,0.5));
}



double getWeights(std::string t_kernel, 
                  double t_freq, 
                  arma::vec t_wBeta)
{
  if(t_wBeta.size() != 2)
    Rcpp::stop("The size of argument t_wBeta should be 2.");
  
  double weights;
  if(t_kernel == "linear")
    weights = 1;
  
  if(t_kernel == "linear.weighted"){
    Rcpp::NumericVector freq = {t_freq};
    Rcpp::NumericVector temp = Rcpp::dbeta(freq, t_wBeta(0), t_wBeta(1));
    weights = temp(0);
  }
  
  return weights;
}

void imputeGeno(arma::vec& t_GVec, 
                double t_altFreq, 
                std::vector<uint32_t> t_indexForMissing,
                std::string t_imputeMethod) 
{
  int nMissing = t_indexForMissing.size();
  
  double imputeG = 0;
  
  if(t_imputeMethod == "mean")
    imputeG = 2 * t_altFreq;
  
  if(t_imputeMethod == "none")
    imputeG = arma::datum::nan;
  
  if(t_imputeMethod == "bestguess")
    imputeG = std::round(2 * t_altFreq);
  
  for(int i = 0; i < nMissing; i++){
    uint32_t index = t_indexForMissing.at(i);
    t_GVec.at(index) = imputeG;
  }
}

// used in Main.cpp::mainMarkerInCPP
bool imputeGenoAndFlip(arma::vec& t_GVec, 
                       double & t_altFreq,
		       double & t_altCount, 
                       std::vector<uint32_t> & t_indexForMissing,
                       std::string t_impute_method,
		       double t_dosage_zerod_cutoff,
		       double t_dosage_zerod_MAC_cutoff, 
		       double & t_MAC,
		       std::vector<uint> & t_indexZero,
		       std::vector<uint> & t_indexNonZero)   
{
  bool flip = false;
  t_indexNonZero.clear();
  t_indexZero.clear();
  int nMissing = t_indexForMissing.size();
  uint dosagesSize = t_GVec.size();
  double imputeG = 0;
  if(t_altFreq > 0.5){
    flip = true;
    // BIOMNI-OPT it6: in-place flip (was: arma temp + assign = 4 passes over
    // the vector). Identical per-element FP op (2.0 - x), bitwise-identical.
    double * gflip = t_GVec.memptr();
    for(uint i = 0; i < dosagesSize; i++) gflip[i] = 2.0 - gflip[i];
    t_altFreq = 1 - t_altFreq; 
  }

if(nMissing > 0){

switch(string_to_case.at(t_impute_method)) {
  case 1:
    imputeG = std::round(2 * t_altFreq);
    //std::cout << "t_impute_method " << t_impute_method << std::endl;
    break;
  case 2:
    imputeG = 2 * t_altFreq;
    //std::cout << "t_impute_method " << t_impute_method << std::endl;
    break;
  case 3:
    imputeG = 0;
    //std::cout << "t_impute_method " << t_impute_method << std::endl;
    break;
}


  for(int i = 0; i < nMissing; i++){
    uint32_t j = t_indexForMissing.at(i);
    t_GVec.at(j) = imputeG;
  }

 t_MAC = t_MAC + imputeG * nMissing;

}

  
  if(t_dosage_zerod_cutoff > 0){ 
    if(t_MAC <= t_dosage_zerod_MAC_cutoff){
      t_GVec.clean(t_dosage_zerod_cutoff);
    }	  
  } 

  //if(nMissing > 0 || t_dosage_zerod_cutoff > 0){
     t_altCount = arma::sum(t_GVec);
     t_altFreq = t_altCount / (2*dosagesSize);
     if(flip){
	t_altFreq = 1 - t_altFreq;
        t_altCount = 2*dosagesSize - t_altCount;	
	//t_altCount = 2 * t_altFreq * dosagesSize;
     }	     
  //}

 // BIOMNI-OPT it6: branchless raw-pointer split. The push_back loop pays a
 // data-dependent branch (~50% mispredict for common variants) plus a
 // capacity check per element; this does two unconditional stores and
 // advances one cursor. Resulting vector contents are identical.
 t_indexZero.resize(dosagesSize);
 t_indexNonZero.resize(dosagesSize);
 {
	unsigned int * pz = t_indexZero.data();
	unsigned int * pnz = t_indexNonZero.data();
	const double * gp = t_GVec.memptr();
	size_t nz_cnt = 0, nnz_cnt = 0;
	for(unsigned int i = 0; i < dosagesSize; i++){
		bool isz = (gp[i] == 0);
		pz[nz_cnt] = i;
		pnz[nnz_cnt] = i;
		nz_cnt += isz;
		nnz_cnt += !isz;
	}
	t_indexZero.resize(nz_cnt);
	t_indexNonZero.resize(nnz_cnt);
 }

  return flip;
}


// BIOMNI-OPT it16: arma::uvec-output overload (see UTIL.hpp). Body identical
// to the std::vector version except the split pass, which fills the arma
// vectors in place: set_size reuses n_alloc across markers (arma 15 init_warm
// keeps allocation when new_n_elem <= n_alloc) and never value-initializes,
// removing both the conv_to copy and the std::vector grow-zeroing.
bool imputeGenoAndFlip(arma::vec& t_GVec,
                       double & t_altFreq,
		       double & t_altCount,
                       std::vector<uint32_t> & t_indexForMissing,
                       std::string t_impute_method,
		       double t_dosage_zerod_cutoff,
		       double t_dosage_zerod_MAC_cutoff,
		       double & t_MAC,
		       arma::uvec & t_indexZero,
		       arma::uvec & t_indexNonZero)
{
  bool flip = false;
  int nMissing = t_indexForMissing.size();
  uint dosagesSize = t_GVec.size();
  double imputeG = 0;
  if(t_altFreq > 0.5){
    flip = true;
    double * gflip = t_GVec.memptr();
    for(uint i = 0; i < dosagesSize; i++) gflip[i] = 2.0 - gflip[i];
    t_altFreq = 1 - t_altFreq;
  }

if(nMissing > 0){

switch(string_to_case.at(t_impute_method)) {
  case 1:
    imputeG = std::round(2 * t_altFreq);
    break;
  case 2:
    imputeG = 2 * t_altFreq;
    break;
  case 3:
    imputeG = 0;
    break;
}


  for(int i = 0; i < nMissing; i++){
    uint32_t j = t_indexForMissing.at(i);
    t_GVec.at(j) = imputeG;
  }

 t_MAC = t_MAC + imputeG * nMissing;

}


  if(t_dosage_zerod_cutoff > 0){
    if(t_MAC <= t_dosage_zerod_MAC_cutoff){
      t_GVec.clean(t_dosage_zerod_cutoff);
    }
  }

     t_altCount = arma::sum(t_GVec);
     t_altFreq = t_altCount / (2*dosagesSize);
     if(flip){
	t_altFreq = 1 - t_altFreq;
        t_altCount = 2*dosagesSize - t_altCount;
     }

 t_indexZero.set_size(dosagesSize);
 t_indexNonZero.set_size(dosagesSize);
 {
	arma::uword * pz = t_indexZero.memptr();
	arma::uword * pnz = t_indexNonZero.memptr();
	const double * gp = t_GVec.memptr();
	size_t nz_cnt = 0, nnz_cnt = 0;
	for(unsigned int i = 0; i < dosagesSize; i++){
		bool isz = (gp[i] == 0);
		pz[nz_cnt] = i;
		pnz[nnz_cnt] = i;
		nz_cnt += isz;
		nnz_cnt += !isz;
	}
	// it16: shrink preserving contents. arma::set_size() to a new size
	// <= arma_config::mat_prealloc RELEASES the heap buffer and switches to
	// the uninitialized inline mem_local buffer (arma 15 init_warm small
	// path), discarding the values just written. For tiny counts, copy the
	// filled prefix into a fresh exact-sized vector instead (<= 16 elems).
	if(nz_cnt <= (size_t)arma::arma_config::mat_prealloc && nz_cnt != (size_t)dosagesSize){
	  t_indexZero = arma::uvec(pz, (arma::uword)nz_cnt, true, false);
	}else{
	  t_indexZero.set_size(nz_cnt);
	}
	if(nnz_cnt <= (size_t)arma::arma_config::mat_prealloc && nnz_cnt != (size_t)dosagesSize){
	  t_indexNonZero = arma::uvec(pnz, (arma::uword)nnz_cnt, true, false);
	}else{
	  t_indexNonZero.set_size(nnz_cnt);
	}
 }

  return flip;
}


double getInnerProd(arma::mat& x1Mat, arma::mat& x2Mat)
{
  double innerProd = arma::accu(x1Mat % x2Mat);
  return(innerProd);
}

arma::vec getTime(){
  arma::vec Time(2, arma::fill::zeros);
  struct timeval time;
  Time(0) = 0;
  if(!gettimeofday(&time,NULL))
    Time(0) = (double)time.tv_sec + (double)time.tv_usec * .000001;
  Time(1) = (double)clock() / CLOCKS_PER_SEC;
  return Time;
}

void printTime(arma::vec t1, arma::vec t2, std::string message){
  double wallTime = t2(0) - t1(0);
  double cpuTime = t2(1) - t1(1);
  if(wallTime < 60){
    Rprintf ("It took %f seconds (%f CPU seconds) to %s.\n",
             wallTime, cpuTime, message.c_str());
  }else if(wallTime < 3600){
    Rprintf ("It took %f minutes (%f CPU minutes) to %s.\n",
             wallTime/60, cpuTime/60, message.c_str());
  }else{
    Rprintf ("It took %f hours (%f CPU hours) to %s.\n",
             wallTime/3600, cpuTime/3600, message.c_str());
  }
}

double getinvStd(double t_freq)
{
  double Std = sqrt(2 * t_freq * (1-t_freq));
  if(Std == 0)
    return 0;
  else
    return 1/Std;
}

// [[Rcpp::export]]
double sum_arma1(arma::vec& X) {
    double sum = 0;
    for (uint i = 0; i < X.n_elem; ++i) {
        if (arma::is_finite(X(i)))
            sum += X(i);
    }
    return sum;
}


// [[Rcpp::export]]
double add_logp(double p1, double p2)
{
        using namespace Rcpp;
        p1 = -std::abs(p1);
        p2 = -std::abs(p2);
        double maxp = std::max(p1,p2);
        double  minp = std::min(p1,p2);
        double result = maxp+std::log(1+std::exp(minp-maxp));
        return(result);
}
