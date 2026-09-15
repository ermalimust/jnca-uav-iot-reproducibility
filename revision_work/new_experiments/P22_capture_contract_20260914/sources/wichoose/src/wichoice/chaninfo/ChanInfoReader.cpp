/**
 * Implements class tha reads channel information associated with wifi interfaces.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include <fstream>   // std::ifstream
#include <sstream>   // std::stringstream
#include <string>    // std::string
#include <tuple>     // std::forward_as_tuple
#include <utility>   // std::piecewise_construct
#include <exception> // std::exception

#include "../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../util/conf/Network.hpp"      // type NetworkId

#include "ChanInfoReader.hpp" // class ChanInfoReader

#define WI_INFO_FNAME "/proc/net/wireless"

/**
 * Empty constructor.
 */
ChanInfoReader::ChanInfoReader() : chanInfoMap(), netIdMap() {
  
  // read networks
  WcConfigFile configFile;
  NetworkMap netMap;
  configFile.networks("wichoice", netMap);
  
  // initialize both netIdMap and chanInfoMap
  for (auto& kvp : netMap) {
    const NetworkId netId = kvp.first;
    Network const& net = kvp.second;
    
    this->netIdMap.emplace(net.iface, netId); // add iface -> net id entry
  
    // add net id -> chan info entry
    this->chanInfoMap.emplace(std::piecewise_construct,
                              std::forward_as_tuple(netId),
                              std::forward_as_tuple(RSSI_DISCONNECTED));
  }
}

enum ReadState { HEADER, BODY}; // for file-reading state machine

/**
 * Update the associated channel information by reading the currently-available values.
 * Doesn't return anything, just the update side effect.
 */
void ChanInfoReader::updateChanInfo() {
  
  // default all rssi values to unknown
  for (auto& kvp : this->chanInfoMap) {
    ChanInfo& chanInfo = kvp.second;
    chanInfo.rssi = RSSI_DISCONNECTED;
  }

  std::ifstream wifile(WI_INFO_FNAME, std::ifstream::in);

  // does the file exist and can we read from it?
  if (!wifile.good()) {
    std::stringstream ss;
    ss << "ChanInfoReader::updateChanInfo() message: can't open file " <<
          WI_INFO_FNAME << ". Can't read RSSI values." << std::endl;
    LOG_MSG(ss.str().c_str());
    return;
  }

  // value-reading helper variables
  std::string line;  // individual line buffer
  std::string taway; // throwaway token-consumer string
  std::string iface;
  int rssi;

  ReadState state = ReadState::HEADER; // start with
  unsigned hlineCtr = 0; // header line
  
  try {
    // continue while we're not at the end of the file
    // read file line by line
    while (std::getline(wifile, line)) {

      switch (state) { // reading state machine
        case ReadState::HEADER: {
          if (++hlineCtr == 2) state = BODY; // header has two lines
          break;
        }
        case ReadState::BODY: {
          // convert to stream and listen to exceptions
          std::istringstream iss(line);
          iss.exceptions(std::istringstream::failbit |
                         std::istringstream::badbit);
          
          // line format: "iface: status link-q level-q noise-q ..."
          iss >> iface >> taway >> taway >> rssi;
          
          if (rssi != 0 && rssi != -256) { // zero and -256 mean no comm
            // strip the colon suffix out of the interface's name
            iface.pop_back();
          
            // is this an interface we care about?
            auto itr = this->netIdMap.find(iface);
            if (itr != this->netIdMap.end()) { // yes, so record info
              const NetworkId netId = itr->second;
              this->chanInfoMap[netId].rssi = rssi; // just update the value
            }
          } // rssi != 0 if end
        } // case BODY end
      } // state machine end
    } // while getline end
    
    // log the rssi values we've read
    std::stringstream ss;
    ss << "ChanInfoReader read RSSIs: ";
    for (auto& kvp : this->chanInfoMap) {
      const uint16_t ifaceId = kvp.first;
      ChanInfo const& chanInfo = kvp.second;
      ss << "iface " << ifaceId << "=" << chanInfo.rssi << " ";
    }
    LOG_VERBOSE(ss.str().c_str());
    
  } catch (std::exception const& e) { // problem opening or reading from file
      std::stringstream ss;
      ss << "ChanInfoReader::updatedChanInfo() file reading error: "
        << e.what()
        << ", line: " << line << std::endl;
      LOG_FATAL_EXIT(ss.str().c_str());
  }
  
  wifile.close(); // file no longer needed
}

/**
 * Returns a (const) reference to the channel info map associated with the reader.
 *
 * @return a reference to the channel info map associated with the reader.
 */
ChanInfoMap const& ChanInfoReader::getChanInfoMap() const {
  return this->chanInfoMap;
}
