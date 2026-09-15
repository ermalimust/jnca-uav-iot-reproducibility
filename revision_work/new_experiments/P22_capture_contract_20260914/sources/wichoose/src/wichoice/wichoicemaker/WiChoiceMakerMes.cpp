/**
 * Implements  a concrete class that chooses wifi interfaces based on measurement-based data.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "WiChoiceMakerMes.hpp" // class WiChoiceMakerMes

/**
 * Constructor method.
 *
 * @param nperfDb: historical network performance database to use when picking an interface.
 * @param nperfSumStrat: network performance summarization strategy to use.
 */
WiChoiceMakerMes::WiChoiceMakerMes(NperfDb& nperfDb,
                                   NperfSumStrat const& nperfSumStrat) :
                                   WiChoiceMaker(nperfDb, nperfSumStrat, "Mes"),
                                   wiChoiceWriterMes() { }
  
/**
 * Compute and return the expected throughput from the provided performance summaries using measured
 * data.
 *
 * @param nperfSums: the performance summaries to compute a mean from.
 * @return the mean throughput from the summaries, from measurements.
 */
uint64_t WiChoiceMakerMes::getExpTput(NperfSums const& nperfSums) {
  // take measurement-based component and delegate to single-summary version
  return this->nperfSumStrat.getExpTput(nperfSums.mes);
}

/**
 * Returns the object that should be used to write new wifi interface choices to shared memory.
 *
 * @return object to use to write new wifi interface choices to shared memory.
 */
WiChoiceWriter& WiChoiceMakerMes::getWiChoiceWriter() {
  return this->wiChoiceWriterMes;
}
