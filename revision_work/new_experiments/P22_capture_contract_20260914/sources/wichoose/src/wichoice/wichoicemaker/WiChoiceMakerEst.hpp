/**
 * Defines a concrete class to choose wifi interfaces based on estimate-based data.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WI_CHOICE_MAKER_EST_HPP__
#define WI_CHOICE_MAKER_EST_HPP__

#include <cstdint> // uint*_t

#include "../shm/WiChoiceWriterEst.hpp" // class WiChoiceWriterEst
#include "database/NperfDb.hpp"         // class NperfDb, struct NperfSums
#include "nperfsum/NperfSum.hpp"        // struct NperfSums
#include "nperfsum/NperfSumStrat.hpp"   // class NperfSumStrat
#include "WiChoiceMaker.hpp"            // class WiChoiceMaker

class WiChoiceMakerEst : public WiChoiceMaker {

public:
  /**
   * Constructor method.
   *
   * @param nperfDb: historical network performance database to use when picking an interface.
   * @param nperfSumStrat: network performance summarization strategy to use.
   */
  WiChoiceMakerEst(NperfDb& nperfDb, NperfSumStrat const& nperfSumStrat);
  
protected:
  /**
   * Compute and return the expected throughput from the provided performance summaries using
   * estimated data.
   *
   * @param nperfSums: the performance summaries to compute a mean from.
   * @return the mean throughput from the summaries, from measurements.
   */
  uint64_t getExpTput(NperfSums const& nperfSums) override;
  
  /**
   * Returns the object that should be used to write new wifi interface choices to shared memory.
   *
   * @return object to use to write new wifi interface choices to shared memory.
   */
  WiChoiceWriter& getWiChoiceWriter() override;
  
private:
  WiChoiceWriterEst wiChoiceWriterEst;
};

#endif // WI_CHOICE_MAKER_EST_HPP__
