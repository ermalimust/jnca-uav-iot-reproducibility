/**
 * Defines an abstract interface for a strategy to summarize performance data.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef NPERF_SUM_STRAT_HPP__
#define NPERF_SUM_STRAT_HPP__

#include <cstdint> // uint*_t

#include "../../../dtransfer/tput/Tput.hpp" // type Tput
#include "NperfSum.hpp"                     // struct NperfSum

class NperfSumStrat {

public:
  /**
   * Destructor. Needs to be made virtual on abstract classes to ensure proper destruction.
   */
  virtual ~NperfSumStrat();
  
  /**
   * Adds a throughput sample value to a performance summary.
   *
   * @param tput: the throughput sample to be added.
   * @param nperfSum: the performance summary to add the sample to.
   */
  virtual void addTputSamp(const Tput tput, NperfSum& nperfSum) const = 0;
  
  /**
   * Computes and returns the expected throughput for the performance summary passed as as argument.
   *
   * @param nperfSum: the performance summary to compute the expected throughput for.
   * @return the computed expected throughput.
   */
  virtual uint64_t getExpTput(NperfSum const& nperfSum) const = 0;
  
protected:
  /**
   * Empty constructor.
   */
  NperfSumStrat();
};

#endif // NPERF_SUM_STRAT_HPP__
