/**
 * Implements  a concrete class that chooses wifi interfaces based on estimate-based data.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "WiChoiceMakerEst.hpp" // class WiChoiceMakerEst

/**
 * Constructor method.
 *
 * @param nperfDb: historical network performance database to use when picking an interface.
 * @param nperfSumStrat: network performance summarization strategy to use.
 */
WiChoiceMakerEst::WiChoiceMakerEst(NperfDb& nperfDb,
                                   NperfSumStrat const& nperfSumStrat) :
                                   WiChoiceMaker(nperfDb, nperfSumStrat, "Est"),
                                   wiChoiceWriterEst() { }

/**
 * Compute and return the expected throughput from the provided performance summaries using
 * estimated data.
 *
 * @param nperfSums: the performance summaries to compute a mean from.
 * @return the mean throughput from the summaries, from estimates.
 */
uint64_t WiChoiceMakerEst::getExpTput(NperfSums const& nperfSums) {
  // take estimate-based component and delegate to single-summary version
  return this->nperfSumStrat.getExpTput(nperfSums.est);
}

/**
 * Returns the object that should be used to write new wifi interface choices to shared memory.
 *
 * @return object to use to write new wifi interface choices to shared memory.
 */
WiChoiceWriter& WiChoiceMakerEst::getWiChoiceWriter() {
  return this->wiChoiceWriterEst;
}
