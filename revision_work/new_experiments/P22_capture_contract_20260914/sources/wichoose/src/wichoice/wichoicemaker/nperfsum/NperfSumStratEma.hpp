/**
 * Defines a class to summarize performance data using a simple exponential moving average.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef NPERF_SUM_STRAT_EMA_HPP__
#define NPERF_SUM_STRAT_EMA_HPP__

#include <cstdint> // uint*_t

#include "../../../dtransfer/tput/Tput.hpp" // type Tput
#include "NperfSum.hpp"                     // struct NperfSum
#include "NperfSumStrat.hpp"                // class NperfSumStrat

class NperfSumStratEma : public NperfSumStrat {

public:
  /**
   * Constructor method. The newSampWeight arguments, also known as alpha, tells us how much
   * weight to assign to the new sample, when computing the exponential moving average.
   * Throws exception if newSampleWeight argument is not in [0,1].
   *
   * @param newSampWeight: weight to assign to an incoming new sample.
   */
  NperfSumStratEma(const double newSampWeight);
  
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
  
private:
  const double newSampWeight;
};

#endif // NPERF_SUM_STRAT_EMA_HPP__
