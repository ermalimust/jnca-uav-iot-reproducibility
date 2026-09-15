/**
 * Defines a class to summarize performance data using a simple arithmetic mean.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef NPERF_SUM_STRAT_AM_HPP__
#define NPERF_SUM_STRAT_AM_HPP__

#include <cstdint> // uint*_t

#include "../../../dtransfer/tput/Tput.hpp" // type Tput
#include "NperfSum.hpp"                     // struct NperfSum
#include "NperfSumStrat.hpp"                // class NperfSumStrat

class NperfSumStratAm : public NperfSumStrat {

public:
  /**
   * Empty constructor.
   */
  NperfSumStratAm();
  
  /**
   * Adds a throughput sample value to a performance summary.
   *
   * @param tput: the throughput sample to be added.
   * @param nperfSum: the performance summary to add the sample to.
   */
  void addTputSamp(const Tput tput, NperfSum& nperfSum) const override;
  
  /**
   * Computes and returns the expected throughput for the performance summary passed as as argument
   * as an ex.
   *
   * @param nperfSum: the performance summary to compute the expected throughput for.
   * @return the computed expected throughput.
   */
  uint64_t getExpTput(NperfSum const& nperfSum) const override;
};

#endif // NPERF_SUM_STRAT_AM_HPP__
