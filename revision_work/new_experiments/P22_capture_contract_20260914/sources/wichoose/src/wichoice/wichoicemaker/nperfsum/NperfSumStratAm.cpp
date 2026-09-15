/**
 * Implement a class that summarizes performance data using a simple arithmetic mean
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "NperfSumStratAm.hpp" // class NperfSumStratEma

/**
 * Empty constructor.
 */
NperfSumStratAm::NperfSumStratAm() { }

/**
 * Adds a throughput sample value to a performance summary.
 *
 * @param tput: the throughput sample to be added.
 * @param nperfSum: the performance summary to add the sample to.
 */
void NperfSumStratAm::addTputSamp(const Tput tput, NperfSum& nperfSum) const {
  nperfSum.tputSum += tput;
  nperfSum.nentries++;
}

/**
 * Computes and returns the expected throughput for the performance summary passed as as argument
 * as an ex.
 *
 * @param nperfSum: the performance summary to compute the expected throughput for.
 * @return the computed expected throughput.
 */
uint64_t NperfSumStratAm::getExpTput(NperfSum const& nperfSum) const {
  return nperfSum.nentries > 0 ? // guard against lack of data
         nperfSum.tputSum / nperfSum.nentries : 0;
}
