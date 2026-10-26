
// All C++ codes related to PLINK file manipulation

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

#include <boost/algorithm/string.hpp>
#include "PLINK.hpp"
#include "UTIL.hpp"

#include <RcppArmadilloExtensions/sample.h> // sample
#include <string>

// make a global variable for future usage
// static PLINK::PlinkClass* ptr_gPLINKobj = NULL;

// ptr_gPLINKobj = new PLINK::PlinkClass(t_bimFile,
//                                       t_famFile,
//                                       t_bedFile,
//                                       t_SampleInModel);

namespace PLINK {

PlinkClass::PlinkClass(std::string t_bimFile,
                       std::string t_famFile,
                       std::string t_bedFile,
                       //std::vector<std::string> t_SampleInModel,
                       std::string t_AlleleOrder)
{
  setPlinkobj(t_bimFile, t_famFile, t_bedFile);
  //setPosSampleInPlink(t_SampleInModel);
  m_AlleleOrder = t_AlleleOrder;
}



// set PlinkClass by reading PLINK files
void PlinkClass::setPlinkobj(std::string t_bimFile,
                             std::string t_famFile,
                             std::string t_bedFile)
{
  m_bimFile = t_bimFile;
  m_famFile = t_famFile;
  m_bedFile = t_bedFile;
  
  // setChrMaps();
  readBimFile();
  readFamFile();
  
  m_fin = fopen(t_bedFile.c_str(), "rb");
  // BIOMNI-OPT it16: 4MB stdio buffer (default is st_blksize ~4KB). Marker
  // reads are sequential within a chunk, so this cuts read() syscalls ~1000x.
  // NULL buffer => glibc allocates; bytes read are identical.
  setvbuf(m_fin, NULL, _IOFBF, 4*1024*1024);
  char magicNumber_2[65536];
  fread(magicNumber_2, 2, 1, m_fin);
  char magicNumber3;
  fread(&magicNumber3, 1, 1, m_fin);
  fseek(m_fin, 3, SEEK_SET);    

/* 
  m_ibedFile.open(m_bedFile.c_str(), std::ios::binary);
  m_ibedFile.seekg(2);
  char magicNumber3;
  m_ibedFile.read(&magicNumber3, 1);
*/

  if(magicNumber3 != 1)
    Rcpp::stop("The third magic number of the plink bed file is not 00000001. Please use SNP-major plink (plink version >= 1.9) files.");
}

void PlinkClass::readBimFile()
{
  std::cout << "Reading bim file...." << std::endl;
  std::ifstream bim(m_bimFile);
  m_M0 = 0;
  std::string line;
  
  while(getline(bim, line)){
    m_M0++;
    std::vector<std::string> line_elements;
    boost::split(line_elements, line, boost::is_any_of("\t "));
    boost::replace_all(line_elements[line_elements.size() - 1], "\r", "");
    
    m_chr.push_back(line_elements[0]);
    m_MarkerInPlink.push_back(line_elements[1]);
    m_gd.push_back(std::stof(line_elements[2]));
    m_pd.push_back(std::stoi(line_elements[3]));
    std::transform(line_elements[4].begin(), line_elements[4].end(), line_elements[4].begin(), toupper);
    std::transform(line_elements[5].begin(), line_elements[5].end(), line_elements[5].begin(), toupper);
    m_alt.push_back(line_elements[4]);  // allele 1, usually minor allele, alt-first
    m_ref.push_back(line_elements[5]);  // allele 2, usually major allele
  }
  m_M = m_M0;
}

void PlinkClass::readFamFile()
{
  std::cout << "Reading fam file...." << std::endl;
  std::ifstream fam(m_famFile);
  m_N0 = 0;
  std::string line;
  while(getline(fam, line)){
    m_N0 ++;
    std::vector<std::string> line_elements;
    boost::split(line_elements, line, boost::is_any_of("\t "));
    m_SampleInPlink.push_back(line_elements[1]);  // put IID to m_SampleInPlink
  }
  m_N = m_N0;
  m_numBytesofEachMarker0 = (m_N0 + 3) / 4;
  m_OneMarkerG4.reserve(m_numBytesofEachMarker0);  
  m_OneMarkerG4.resize(m_numBytesofEachMarker0);
} 


void PlinkClass::setPosSampleInPlink(std::vector<std::string> & t_SampleInModel)
{
  std::cout << "Setting position of samples in PLINK files...." << std::endl;
  m_N = t_SampleInModel.size();
  m_numBytesofEachMarker = (m_N + 3) / 4;
  
  // convert from std::vector<std::string> to Rcpp::CharacterVector
  Rcpp::CharacterVector SampleInModel(m_N);
  for(uint32_t i = 0; i < m_N; i++)
    SampleInModel(i) = t_SampleInModel.at(i);
  
  // convert from std::vector<std::string> to Rcpp::CharacterVector
  uint32_t N_plink = m_SampleInPlink.size();
  Rcpp::CharacterVector SampleInPlink(N_plink);
  for(uint32_t i = 0; i < N_plink; i++)
    SampleInPlink(i) = m_SampleInPlink.at(i);
    
  // Rcpp::match is much faster than loop
  Rcpp::IntegerVector posSampleInPlink = Rcpp::match(SampleInModel, SampleInPlink);
  m_posSampleInPlink.resize(m_N);
  for(uint32_t i = 0; i < m_N; i++){
    if(Rcpp::IntegerVector::is_na(posSampleInPlink.at(i)))
      Rcpp::stop("At least one subject requested is not in Plink file.");
    m_posSampleInPlink.at(i) = posSampleInPlink.at(i) - 1;   // convert "starting from 1" to "starting from 0"
    //std::cout << "m_posSampleInPlink.at(i) " << i << " " << m_posSampleInPlink.at(i) << std::endl;
     }



  // BIOMNI-OPT it13: detect identity sample map and prebuild decode LUTs.
  m_fastDecode = (m_N % 4 == 0);
  if(m_fastDecode){
    for(uint32_t k = 0; k < m_N; k++){
      if(m_posSampleInPlink[k] != k){ m_fastDecode = false; break; }
    }
  }
  if(m_fastDecode){
    const std::vector<int8_t> * gm = nullptr;
    if(m_AlleleOrder == "alt-first"){ gm = &m_genoMaps_alt_first; }
    else if(m_AlleleOrder == "ref-first"){ gm = &m_genoMaps_ref_first; }
    if(gm == nullptr){
      m_fastDecode = false;
    }else{
      for(int b = 0; b < 256; b++){
        uint64_t c = 0; uint8_t mm = 0;
        for(int s = 0; s < 4; s++){
          int v = (b >> (2*s)) & 3;
          c += (uint64_t)1 << (16*v);  // += not |=: duplicate codes within a byte must accumulate (lane max 4, no carry)
          if(v == MISSING){ mm |= (1 << s); }
          // exact original conversion chain: int8_t -> size_t -> double
          m_lutD_true[b][s] = (double)(size_t)(*gm)[v];
          m_lutD_false[b][s] = (double)v;
        }
        m_lutCnt[b] = c; m_lutMiss[b] = mm;
      }
    }
  }
}

// std::vector<uint32_t> PlinkClass::getPosMarkerInPlink(std::vector<std::string> t_MarkerReqstd)
// {
//   int M = t_MarkerReqstd.size();
//   std::vector<uint32_t> posMarkerInPlink;
//   for(int i = 0; i < M; i++){
//     std::string marker = t_MarkerReqstd.at(i);
//     auto pos = std::find(m_MarkerInPlink.begin(), m_MarkerInPlink.end(), marker);
//     if(pos != m_MarkerInPlink.end()){
//       posMarkerInPlink.push_back(pos - m_MarkerInPlink.begin());
//     }else{
//       Rcpp::warning("Marker %s is not found in plink file.", marker);
//     }
//   }
//   return posMarkerInPlink;
// }

void PlinkClass::closegenofile(){
        fclose(m_fin);
	//m_ibedFile.close();
}


void PlinkClass::getOneMarker(uint64_t & t_gIndex_prev,
				uint64_t & t_gIndex,        // different meanings for different genoType
                                   std::string& t_ref,       // REF allele
                                   std::string& t_alt,       // ALT allele (should probably be minor allele, otherwise, computation time will increase)
                                   std::string& t_marker,    // marker ID extracted from genotype file
                                   uint32_t& t_pd,           // base position
                                   std::string& t_chr,       // chromosome
                                   double& t_altFreq,        // frequency of ALT allele
                                   double& t_altCounts,      // counts of ALT allele
                                   double& t_missingRate,    // missing rate
                                   double& t_imputeInfo,     // imputation information score, i.e., R2 (all 1 for PLINK)
                                   bool & t_isOutputIndexForMissing,               // if true, output index of missing genotype data
                                   std::vector<uint>& t_indexForMissing,     // index of missing genotype data
                                   bool & t_isOnlyOutputNonZero,                   // if true, only output a vector of non-zero genotype. (NOTE: if ALT allele is not minor allele, this might take much computation time)
                                   std::vector<uint>& t_indexForNonZero,     // only used when t_isOnlyOutputNonZero = TRUE
                                   bool & t_isTrueGenotype,    // only used in PLINK. check m_genoMaps for details about the genotype mapping in PLINK.
				   arma::vec & OneMarkerG1)
				   //std::vector<double>& OneMarkerG1)
{		
  
  //std::vector<double> OneMarkerG1;
 //arma::vec timeoutput1 = getTime(); 
  // t_isTrueGenotype = FALSE is used only when calculating GRM
  if(!t_isTrueGenotype){
    if(t_isOutputIndexForMissing)
      Rcpp::stop("Check PlinkClass::getOneMarker, if t_isTrueGenotype = FALSE, then t_isOutputIndexForMissing should be FALSE.");
    if(t_isOnlyOutputNonZero)
      Rcpp::stop("Check PlinkClass::getOneMarker, if t_isTrueGenotype = FALSE, then t_isOnlyOutputNonZero should be FALSE.");
  }
 
  // BIOMNI-OPT it19: serve marker bytes from a 16MB prefetch buffer instead of
  // one buffered fread per marker. glibc refilled its stdio buffer with 4KB
  // read() syscalls (~122k syscalls for a 500MB bed) at ~10us each under
  // production interleaving; bulk 16MB freads bypass the stdio buffer and cut
  // fetch time ~15x. The byte stream is identical, so all downstream values
  // are bitwise-identical. A short read at EOF copies only the available
  // prefix and leaves the tail of m_OneMarkerG4 stale, exactly matching the
  // original fread's partial-fill semantics.
  {
    const uint64_t nb = m_numBytesofEachMarker0;
    const uint64_t markerOff = 3 + nb * (uint64_t)t_gIndex;
    if(markerOff < m_markerBufStart || markerOff + nb > m_markerBufStart + m_markerBufLen){
      const uint64_t CAP = 16u << 20;
      if(m_markerBuf.size() < CAP){ m_markerBuf.resize(CAP); }
      fseek(m_fin, (long)markerOff, SEEK_SET);
      size_t got = fread((char*)m_markerBuf.data(), 1, (size_t)CAP, m_fin);
      m_markerBufStart = markerOff;
      m_markerBufLen = (uint64_t)got;
    }
    uint64_t avail = 0;
    if(markerOff + nb <= m_markerBufStart + m_markerBufLen){ avail = nb; }
    else if(m_markerBufStart + m_markerBufLen > markerOff){ avail = m_markerBufStart + m_markerBufLen - markerOff; }
    if(avail > 0){ memcpy(&m_OneMarkerG4[0], m_markerBuf.data() + (markerOff - m_markerBufStart), (size_t)avail); }
  }

  t_indexForMissing.clear();
  t_indexForNonZero.clear();
  
  t_marker = m_MarkerInPlink[t_gIndex];
  t_pd = m_pd[t_gIndex];
  t_chr = m_chr[t_gIndex];
  
  //std::map<int8_t, int8_t> genoMaps;
  std::vector<int8_t> genoMaps;
  
  if(m_AlleleOrder == "alt-first"){
    t_ref = m_ref[t_gIndex];
    t_alt = m_alt[t_gIndex];

  }
    
  if(m_AlleleOrder == "ref-first"){
    t_ref = m_alt[t_gIndex];
    t_alt = m_ref[t_gIndex];

  }
 //arma::vec timeoutput2 = getTime();
  uint j = 0;
  int counts[] = {0, 0, 0, 0};
  if(m_fastDecode && !t_isOnlyOutputNonZero){
    // BIOMNI-OPT it13: byte-LUT decode (identity sample map, m_N%4==0).
    // Reproduces the original per-sample loop exactly: same stored doubles
    // (int8_t->size_t->double chain baked into the LUTs), same raw counts,
    // same increasing missing-index list. 16-bit count lanes are flushed
    // every 4096 bytes so any m_N works.
    const double (*lutD)[4] = t_isTrueGenotype ? m_lutD_true : m_lutD_false;
    const unsigned char* src = m_OneMarkerG4.data();
    double* dst = OneMarkerG1.memptr();
    uint32_t nb = m_N/4;
    int c0=0, c1=0, c2=0, c3=0;
    size_t b0 = 0;
    while(b0 < nb){
      size_t b1 = b0 + 4096; if(b1 > (size_t)nb){ b1 = nb; }
      uint64_t cnt4 = 0;
      if(t_isOutputIndexForMissing){
        for(size_t b = b0; b < b1; b++){
          unsigned char byte = src[b];
          cnt4 += m_lutCnt[byte];
          uint8_t mm = m_lutMiss[byte];
          if(mm){
            for(int s = 0; s < 4; s++){ if(mm & (1<<s)){ t_indexForMissing.push_back(4*b+s); } }
          }
          const double* row = lutD[byte];
          dst[4*b] = row[0]; dst[4*b+1] = row[1]; dst[4*b+2] = row[2]; dst[4*b+3] = row[3];
        }
      }else{
        for(size_t b = b0; b < b1; b++){
          unsigned char byte = src[b];
          cnt4 += m_lutCnt[byte];
          const double* row = lutD[byte];
          dst[4*b] = row[0]; dst[4*b+1] = row[1]; dst[4*b+2] = row[2]; dst[4*b+3] = row[3];
        }
      }
      c0 += cnt4 & 0xffff; c1 += (cnt4>>16) & 0xffff; c2 += (cnt4>>32) & 0xffff; c3 += (cnt4>>48) & 0xffff;
      b0 = b1;
    }
    counts[0]=c0; counts[1]=c1; counts[2]=c2; counts[3]=c3;
  }else{
    // BIOMNI-OPT it13: genoMaps copy moved here from before the loop --
    // the loop is its only consumer.
    if(m_AlleleOrder == "alt-first"){ genoMaps = m_genoMaps_alt_first; }
    if(m_AlleleOrder == "ref-first"){ genoMaps = m_genoMaps_ref_first; }
  for(uint32_t i = 0; i < m_N; i++)
  {
    auto ind = m_posSampleInPlink[i];             // C++ start from 0
    unsigned char bufferG4 = m_OneMarkerG4[ind/4];    // unsigned char: 1 byte for 4 genotypes (4 samples)
    size_t bufferG1;                                     // int: 1 genotype (1 sample)
    
    // https://www.cog-genomics.org/plink/1.9/formats#bed
    getGenotype(bufferG4, ind%4, bufferG1);          // bufferG4 -> bufferG1
    
    counts[bufferG1]++;

    if(bufferG1 == MISSING && t_isOutputIndexForMissing){
      t_indexForMissing.push_back(i);
    }
    
    
    if(t_isTrueGenotype)
      bufferG1 = genoMaps[bufferG1];


    // BIOMNI-OPT it8: t_indexForNonZero is documented "only used when
    // t_isOnlyOutputNonZero = TRUE". Audited every false-caller (Main.cpp
    // mainMarkerInCPP/mainRegionInCPP/cond, LDmat.cpp, R RareEffect.R): all
    // clear+rebuild it in imputeGenoAndFlip (or never read it) before any use,
    // so building it here was dead work with a mispredicted branch per sample.
    if(t_isOnlyOutputNonZero){
      if(bufferG1 > 0){
        t_indexForNonZero.push_back(i);
        OneMarkerG1[j] = bufferG1;
	      j = j + 1;
      }
    }else{
      OneMarkerG1[i] = bufferG1;
    }
  }
  }

  int numMissing = counts[MISSING];
  
 //arma::vec timeoutput3 = getTime();
  int count = m_N - numMissing;
  t_missingRate = (double)numMissing / (double)m_N;
  t_imputeInfo = 1;
  t_altCounts = (double)(counts[HET] + 2*counts[HOM_ALT]);

  if(count > 0){
        t_altFreq = t_altCounts/ (double)count / 2;
  }else{
        t_altFreq = 0;
  }
  
  // updated on 03/14/2021
  if(m_AlleleOrder == "ref-first"){
    t_altFreq = 1 - t_altFreq;
    t_altCounts = 2 * (double)count * t_altFreq;
  }
 
  if(t_isOnlyOutputNonZero){
    OneMarkerG1.resize(j);
  }
 //arma::vec timeoutput4 = getTime();
  //return OneMarkerG1;
   //  printTime(timeoutput1, timeoutput2, "Unified_getOneMarker 1");
   //  printTime(timeoutput2, timeoutput3, "Unified_getOneMarker 2");
   //  printTime(timeoutput3, timeoutput4, "Unified_getOneMarker 3");
}

// C++ version of which(). Note: start from 0, not 1 
std::vector<unsigned int> whichCPP(std::vector<std::string> & strVec, 
                                   std::string strValue)
{
  std::vector<unsigned int> indexVec;
  for(unsigned int i = 0; i < strVec.size(); i++){
    if(strVec.at(i)==strValue)
      indexVec.push_back(i);
  }
  return(indexVec);
}


}

// make a global variable for future usage
// static PLINK::PlinkClass* ptr_gPLINKobj = NULL;

// ptr_gPLINKobj = new PLINK::PlinkClass(t_bimFile,
//                                       t_famFile,
//                                       t_bedFile,
//                                       t_SampleInModel);


// // [[Rcpp::export]]
// void setPLINKobjInR(std::string t_bimFile,
//                     std::string t_famFile,
//                     std::string t_bedFile,
//                     std::vector<std::string> t_SampleInModel)
// {
//   ptr_gPLINKobj = new PLINK::PlinkClass(t_bimFile,
//                                         t_famFile,
//                                         t_bedFile,
//                                         t_SampleInModel);
//   
//   int n = ptr_gPLINKobj->getN();
//   std::cout << "n:\t" << n << std::endl;
// }



