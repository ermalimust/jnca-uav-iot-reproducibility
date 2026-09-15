/**
 * Defines class to read channel information associated with wifi interfaces.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef CHAN_INFO_READER_HPP__
#define CHAN_INFO_READER_HPP__

#include <string> // std::string
#include <map>    // std::map

#include "ChanInfo.hpp"                // type ChanInfoMap
#include "../../util/conf/Network.hpp" // type NetworkId

class ChanInfoReader {
public:
  /**
   * Empty constructor.
   */
  ChanInfoReader();

  /**
   * Update the associated channel information by reading the currently-available values.
   * Doesn't return anything, just the update side effect.
   */
  void updateChanInfo();

  /**
   * Returns a (const) reference to the channel info map associated with the reader.
   *
   * @return a reference to the channel info map associated with the reader.
   */
  ChanInfoMap const& getChanInfoMap() const;

private:
  ChanInfoMap chanInfoMap;                   // net id -> chan info
  std::map<std::string, NetworkId> netIdMap; // iface name -> net id (helper)
};

#endif // CHAN_INFO_READER_HPP__
