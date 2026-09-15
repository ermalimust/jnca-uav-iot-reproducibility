/**
 * Defines IEEE 802.11ac throughput estimator class.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef TPUT_ESTIMATOR_AC_HPP__
#define TPUT_ESTIMATOR_AC_HPP__

#include "../../../dtransfer/tput/Tput.hpp" // type Tput
#include "../../../gps/GpsInfo.hpp"         // struct GpsInfo
#include "../../chaninfo/ChanInfo.hpp"      // struct ChanInfo
#include "../mobility/ApDistCalculator.hpp" // class ApDistCalculator
#include "TputEstimator.hpp"                // class TputEstimator

class TputEstimatorAc : public TputEstimator {

public:
  /**
   * Empty constructor.
   */
  TputEstimatorAc();

  /**
   * Estimates 802.11n throughput based on the provided mobility and channel state information.
   *
   * @param gpsInfo: current mobility information.
   * @param chanInfo: current channel information.
   * @return the estimated throughput.
   */
  Tput estimateTput(GpsInfo const& gpsInfo, ChanInfo const& chanInfo) override;

private:
  ApDistCalculator apDistCalculator;
};

#endif // TPUT_ESTIMATOR_AC_HPP__
