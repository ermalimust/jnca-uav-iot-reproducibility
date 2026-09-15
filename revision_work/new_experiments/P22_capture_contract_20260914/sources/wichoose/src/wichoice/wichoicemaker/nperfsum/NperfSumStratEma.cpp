/**
 * Implement a class that summarizes performance data using a simple exponential moving average.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <stdexcept>  // std::invalid_argument

#include "NperfSumStratEma.hpp" // class NperfSumStratEma

/**
 * Constructor method. The newSampWeight arguments, also known as alpha, tells us how much
 * weight to assign to the new sample, when computing the exponential moving average.
 * Throws exception if newSampleWeight argument is not in [0,1].
 *
 * @param newSampWeight: weight to assign to an incoming new sample.
 */
NperfSumStratEma::NperfSumStratEma(const double newSampWeight) :
                                                  newSampWeight(newSampWeight) {
  
  if (newSampWeight < 0 || newSampWeight > 1) // problem?
    throw std::invalid_argument("New sample weight must be in [0,1]");
}

/**
 * Adds a throughput sample value to a performance summary.
 *
 * @param tput: the throughput sample to be added.
 * @param nperfSum: the performance summary to add the sample to.
 */
void NperfSumStratEma::addTputSamp(const Tput tput, NperfSum& nperfSum) const {

  if (nperfSum.nentries == 0) nperfSum.tputSum = tput; // first entry
  else // not first entry
    nperfSum.tputSum = ((double)tput) * this->newSampWeight +
                       ((double)nperfSum.tputSum) * (1-this->newSampWeight);

  nperfSum.nentries++; // one more entry
}

/**
 * Computes and returns the expected throughput for the performance summary passed as as argument
 * as an ex.
 *
 * @param nperfSum: the performance summary to compute the expected throughput for.
 * @return the computed expected throughput.
 */
uint64_t NperfSumStratEma::getExpTput(NperfSum const& nperfSum) const {
  return nperfSum.tputSum;
}
