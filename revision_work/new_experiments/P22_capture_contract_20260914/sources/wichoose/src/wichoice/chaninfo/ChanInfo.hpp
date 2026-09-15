/**
 * Defines data structure to store channel information collected for a given interface at a certain point in time.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef CHAN_INFO_HPP__
#define CHAN_INFO_HPP__

#include <map>     // std::map

#include "../../util/conf/Network.hpp" // type NetworkId

#define RSSI_DISCONNECTED -999

struct ChanInfo {
  int rssi; // received signal strength (dBm)
  
  /**
   * Constructor method.
   *
   * @param rssi: received signal strength the channel info should contain. Optional (defaults to
   *              RSSI_DISCONNECTED).s
   */
  ChanInfo(const int rssi = RSSI_DISCONNECTED);
};

typedef std::map<NetworkId, ChanInfo> ChanInfoMap; // net id -> chan info

#endif // CHAN_INFO_HPP__
