/**
 * Implements methods associated with channel information structure ChanInfo.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include "ChanInfo.hpp" // struct ChanInfo

/**
 * Constructor method.
 *
 * @param rssi: received signal strength the channel info should contain. Optional (defaults to
 *              RSSI_DISCONNECTED).
 */
ChanInfo::ChanInfo(const int rssi) : rssi(rssi) { }
