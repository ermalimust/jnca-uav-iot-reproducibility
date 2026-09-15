/**
 * Implements IEEE 802.11ad throughput estimator class.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include <cmath> // std::cos(), std::sin(), std::tanh(), std::log(), std::pow()

#include "../../chaninfo/ChanInfo.hpp" // RSSI_DISCONNECTED

#include "TputEstimatorAd.hpp" // class TputEstimatorAd

/**
 * Empty constructor.
 */
TputEstimatorAd::TputEstimatorAd() { }
  
/**
 * Estimates 802.11ad throughput based on the provided mobility and channel state information.
 *
 * @param gpsInfo: current mobility information.
 * @param chanInfo: current channel information.
 * @return the estimated throughput.
 */
Tput TputEstimatorAd::estimateTput(GpsInfo const& gpsInfo,
                                  ChanInfo const& chanInfo) {
  
  // guard against disconnections
  if (chanInfo.rssi == RSSI_DISCONNECTED) return 0;
  
  // normal calculation
  static const int nusers = 1;
  
  const double speedMs = gpsInfo.speed/3.6; // Km/h to m/s
  
  double tputMbps = 0.7334 * chanInfo.rssi +
                    47.74 * std::sin(speedMs * chanInfo.rssi) -
                    112.6 * std::pow(std::tanh(speedMs), 0.25) -
                    115.8 * std::tanh(std::cos(speedMs)) *
                    std::pow(std::log(nusers), 2) +
                    387.9;

  if (tputMbps < 0) tputMbps = 0;
  const Tput tputBps = tputMbps * 1e6 / 8; // from Mbps to Bps
  
  return tputBps;
}
