/**
 * Defines factory for creating network performance summarizer objects out of the info in the configuration file.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef NPERF_SUM_STRAT_FACTORY_HPP__
#define NPERF_SUM_STRAT_FACTORY_HPP__

#include <memory> // std::unique_ptr

#include "NperfSumStrat.hpp" // class NperfSumStrat

class NperfSumStratFactory {

public:
  /**
   * Creates and returns a NperfSumStrat object out of the info in the system configuration file.
   *
   * @return a NperfSumStrat corresponding to the system's configuration.
   */
  static std::unique_ptr<NperfSumStrat> create();
};

#endif // NPERF_SUM_STRAT_FACTORY_HPP__
