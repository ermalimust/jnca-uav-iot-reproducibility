/**
 * Implements IEEE 802.11ac throughput estimator class.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include <cmath> // std::exp(), std::cos()

#include "../../chaninfo/ChanInfo.hpp" // RSSI_DISCONNECTED

#include "TputEstimatorAc.hpp" // class TputEstimatorAc

/**
 * Empty constructor.
 */
TputEstimatorAc::TputEstimatorAc() : apDistCalculator() { }
  
/**
 * Estimates 802.11ac throughput based on the provided mobility and channel state information.
 *
 * @param gpsInfo: current mobility information.
 * @param chanInfo: current channel information.
 * @return the estimated throughput.
 */
Tput TputEstimatorAc::estimateTput(GpsInfo const& gpsInfo,
                                  ChanInfo const& chanInfo) {
  
  // guard against disconnections
  if (chanInfo.rssi == RSSI_DISCONNECTED) return 0;
  
  // normal calculation
  static const int nusers = 1;
  
  const double apDist = apDistCalculator.apDist(gpsInfo);

  double tputMbps = 1.409 * chanInfo.rssi -
                    2.667 * gpsInfo.speed +
                    44311 * std::cos((double)nusers / chanInfo.rssi) -
                    11.14 * std::cos(nusers) -
                    36.77 * std::pow(apDist, 0.25) -
                    44022;

  if (tputMbps < 0) tputMbps = 0;
  const Tput tputBps = tputMbps * 1e6 / 8; // from Mbps to Bps

  return tputBps;
}
