/**
 * Defines abstract interface for different throughput estimators to implement.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef TPUT_ESTIMATOR_HPP__
#define TPUT_ESTIMATOR_HPP__

#include <vector> // std::vector
#include <memory> // std::unique_ptr

#include "../../../dtransfer/tput/Tput.hpp" // type Tput
#include "../../../gps/GpsInfo.hpp"         // struct GpsInfo
#include "../../chaninfo/ChanInfo.hpp"      // struct ChanInfo

class TputEstimator {

public:
  /**
   * Estimates throughput based on the provided mobility and channel state information.
   *
   * @param gpsInfo: current mobility information.
   * @param chanInfo: current channel information.
   * @return the estimated throughput.
   */
  virtual Tput estimateTput(GpsInfo const& gpsInfo,
                            ChanInfo const& chanInfo) = 0;
};

/**
 * A vector of throughput estimator pointers. It has to be pointers because TputEstimator is an abstract class.
 */
typedef std::vector<std::unique_ptr<TputEstimator>> TputEstimatorPtrVec;

#endif // TPUT_ESTIMATOR_HPP__
