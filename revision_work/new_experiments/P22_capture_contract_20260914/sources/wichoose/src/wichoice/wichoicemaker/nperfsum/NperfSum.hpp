/**
 * Defines structures to represent a network performance summary.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef NPERF_SUM_HPP__
#define NPERF_SUM_HPP__

#include <cstdint> // uint*_t

#include "../../../dtransfer/tput/Tput.hpp" // type Tput

/**
 * Summary of a particular network's performance, aggregated over potentially many samples.
 */
struct NperfSum {
  uint32_t nentries = 0; // number of observations
  uint64_t tputSum = 0;  // sum of all observed tputs

  /**
   * Replaces the contents with contents from the argument object, as long as it has entries.
   *
   * @param src: the performance summary to copy from.
   */
  void replaceWithIfNonZero(NperfSum const& src);

  /**
   * Zeroes out the performance summary on which it is called.
   */
  void reset();
};

/**
 * Aggregate of a network's measured and estimated performance summaries.
 */
struct NperfSums {
  NperfSum est;
  NperfSum mes;

  /**
   * Replaces the contents with contents from the argument object, as long as it has entries.
   *
   * @param src: the performance summaries to copy from.
   */
  void replaceWithIfNonZero(NperfSums const& src);

  /**
   * Zeroes out the performance summaries on which it is called.
   */
  void reset();
};

// table of net performance summaries (dimensions #nets x lookahead)
typedef std::vector<NperfSums> NperfSumVector;
typedef std::vector<NperfSumVector> NperfSumTable;

#endif // NPERF_SUM_HPP__
