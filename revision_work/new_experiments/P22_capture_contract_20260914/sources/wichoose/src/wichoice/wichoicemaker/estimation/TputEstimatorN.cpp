/**
 * Implements IEEE 802.11n throughput estimator class.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include <cmath> // std::exp()

#include "../../chaninfo/ChanInfo.hpp" // RSSI_DISCONNECTED

#include "TputEstimatorN.hpp" // class TputEstimatorN

/**
 * Empty constructor.
 */
TputEstimatorN::TputEstimatorN() { }

/**
 * Estimates 802.11n throughput based on the provided mobility and channel state information.
 *
 * @param gpsInfo: current mobility information.
 * @param chanInfo: current channel information.
 * @return the estimated throughput.
 */
Tput TputEstimatorN::estimateTput(GpsInfo const& gpsInfo,
                                  ChanInfo const& chanInfo) {

  // guard against disconnections
  if (chanInfo.rssi == RSSI_DISCONNECTED) return 0;

  // normal calculation
  static const int nusers = 1;

  double tputMbps = 0.7111 * chanInfo.rssi -
                    2.479 * nusers +
                    11.88 * std::exp(-nusers) +
                    62.02;

  if (tputMbps < 0) tputMbps = 0;
  const Tput tputBps = tputMbps * 1e6 / 8; // from Mbps to Bps

  return tputBps;
}
