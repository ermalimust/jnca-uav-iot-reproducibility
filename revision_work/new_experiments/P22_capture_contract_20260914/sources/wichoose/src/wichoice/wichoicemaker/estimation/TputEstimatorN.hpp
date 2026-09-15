/**
 * Defines IEEE 802.11n throughput estimator class.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef TPUT_ESTIMATOR_N_HPP__
#define TPUT_ESTIMATOR_N_HPP__

#include "../../../dtransfer/tput/Tput.hpp" // type Tput
#include "../../../gps/GpsInfo.hpp"         // struct GpsInfo
#include "../../chaninfo/ChanInfo.hpp"      // struct ChanInfo
#include "TputEstimator.hpp"                // class TputEstimator

class TputEstimatorN : public TputEstimator {

public:
  /**
   * Empty constructor.
   */
  TputEstimatorN();
  
  /**
   * Estimates 802.11n throughput based on the provided mobility and channel state information.
   *
   * @param gpsInfo: current mobility information.
   * @param chanInfo: current channel information.
   * @return the estimated throughput.
   */
  Tput estimateTput(GpsInfo const& gpsInfo, ChanInfo const& chanInfo) override;
};

#endif // TPUT_ESTIMATOR_N_HPP__
