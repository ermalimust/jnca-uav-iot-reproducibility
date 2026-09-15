/**
 * Extends generic ConfigFile class with functionality specific to WiChoose programs.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <linux/socket.h> // AF_INET
#include <arpa/inet.h>    // inet_pton()
#include <cstddef>        // std::size_t
#include <cstdint>        // uint*_t
#include <sstream>        // std::stringstream
#include <string>         // std::string
#include <stdexcept>      // std::invalid_argument
#include <tuple>          // std::forward_as_tuple
#include <utility>        // std::piecewise_construct, std::move

#include "Network.hpp"             // struct Network, types NetworkMap,
                                   // NetworkId
#include "WifiStandardFactory.hpp" // class WifiStandardFactory

#include "WcConfigFile.hpp"        // class WcConfigFile

#define WC_CONFIG_FNAME_DEF "/etc/wichoose.conf"

/**
 * Creates a config object from the filename passed as argument.
 *
 * @param fname name of configuration file.
 */
WcConfigFile::WcConfigFile(std::string const& fname) : ConfigFile(fname) { }

/**
 * Empty constructor creates a config object from the default filename.
 */
WcConfigFile::WcConfigFile() : WcConfigFile(WC_CONFIG_FNAME_DEF) { }

/**
 * Helper that tests whether the argument string represents a valid IP address or not. If it is, it copies it to
 * the destination string passed in as an argument. If there is an error, an error string is set.
 *
 * @param ipStr: the string to be tested.
 * @param dstStr: reference to the destination to which a valid ip string should be copied.
 * @param errorStr: reference to the string that should be set if the validation fails.
 * @return true if the string represents a valid IP address, false otherwise.
 */
bool validateAndCopyIpStr(std::string const& ipStr,
                          std::string& dstStr,
                          std::string& errorStr) {

  struct sockaddr_in saddr{};
  const bool retval = inet_pton(AF_INET, ipStr.c_str(), &(saddr.sin_addr)) == 1;
  
  if (retval) dstStr = ipStr;
  else errorStr = std::string("Invalid IP address ") + ipStr;
  
  return retval;
}

/**
 * Reads all network information from associated configuration file and saves them in the argument map.
 *
 * @param netMap: the map onto which the read networks are to be saved.
 */
void WcConfigFile::networks(NetworkMap& netMap) {

  netMap.clear(); // be sure to start with a clean slate

  const std::string section("networks");
  auto sectionItr = this->sectionMap.find(section);
  if (sectionItr == this->sectionMap.end()) return; // no section, no networks

  EntryMap const& entryMap = sectionItr->second;
  for (const auto& kvp : entryMap) { // for each network entry in section

    Network network; // the network we're parsing right now
    
    std::string errorStr; // for any error msg, defaults to ""
    
    // get network id
    std::string const& netIdStr = kvp.first;
    if (!this->strToUint8(netIdStr, network.id)) // conversion not successful
      errorStr = "Network id must be an integer in [0..255]";
    else if (netMap.count(network.id) != 0)
      errorStr = "Network id must be unique";
    
    if (errorStr != "") {
      this->logError(section, netIdStr, errorStr, "ignoring entry");
      continue; // nothing we can do wo/ a valid id
    }
    
    // id is valid, let's retrieve the remaining info
    bool hasIface=false, hasIpTx=false, hasIpRx=false, hasWiStandard=false;
    
    // tokenize the entry's value, first by comma to separate the pieces
    std::string const& entry = kvp.second;
    std::stringstream sstream(entry);
    // note: ">> std::ws" is used to remove whitespace
    for (std::string epiece; std::getline(sstream >> std::ws, epiece, ',');) {
      std::stringstream esstream(epiece);
      
      // entry piece is composed of a parameter and a value - read them
      const unsigned ninputs = 2;
      std::string strInputs[ninputs];
      for (unsigned i=0; i < ninputs; i++) {
        if (!std::getline(esstream >> std::ws, strInputs[i], ' ')) {
          errorStr = "Every parameter needs a name and a value";
          goto nextPiece;  // nothing else to do here
        }
      } // (param, value) reading end
      
      {
        // process the key-value pair
        std::string const& param = strInputs[0];
        std::string const& value = strInputs[1];
      
        if (param == "iface") {
          network.iface = value;
          hasIface = true;

        } else if (param == "ip-cli") {
          
          if (!(hasIpTx = validateAndCopyIpStr(value, network.ipCli, errorStr)))
            goto nextPiece;
          
        } else if (param == "ip-srv") {
          
          if (!(hasIpRx = validateAndCopyIpStr(value, network.ipSrv, errorStr)))
            goto nextPiece;
          
        } else if (param == "wifi-standard") {
          
          try { // convert wifi standard string to enum
            network.wiStandard = WifiStandardFactory::create(value);
            hasWiStandard = true;
          } catch (std::invalid_argument& e) {
            errorStr = e.what();
            goto nextPiece;
          }
          
        } else { // invalid param
          std::stringstream ss;
          ss << "Invalid parameter " << param
             << " - only iface, iptx, iprx, and wistandard supported";
          errorStr = ss.str();
        }
      } // close variable scope
      
    nextPiece: // will naturally move on to next piece on list
      if (errorStr != "") // do we have an error to report?
        this->logError(section, netIdStr, errorStr, "ignoring entry piece");

    }  // entry piece loop end

    // if everything's in place, it's time to save the network
    if (hasIface && hasIpTx && hasIpRx && hasWiStandard)
      netMap.emplace(std::piecewise_construct,
                     std::forward_as_tuple(network.id),
                     std::forward_as_tuple(std::move(network)));
  } // network loop end
}

/**
 * Reads network information from associated configuration file for section specified as an argument.
 * The networks are stored in the map that is passed in as an argument. The map is indexed by a unique
 * id that is local to the configuration section in question.
 *
 * @param section: name of configuration file section to read from.
 * @param netMap: the map onto which the read networks are to be saved.
 */
void WcConfigFile::networks(std::string const& section, NetworkMap& netMap) {
  
  netMap.clear();    // be sure to start with a clean slate
  uint8_t nnets = 0; // #nets counter

  NetworkMap netMapAll; // start by reading all networks
  this->networks(netMapAll);
  
  // get the networks string as a whole
  const std::string entry = "networks";
  std::string netsStr = this->strValue(section, entry, "" /*defValue*/);
  // tokenize it, first by comma to separate the entries
  // note: ">> std::ws" is used to remove whitespace
  std::stringstream sstream(netsStr);
  
  // go through comma separated network id strings
  for (std::string netIdStr; std::getline(sstream >> std::ws, netIdStr, ',');) {
    
    // convert network id to integer
    NetworkId netId;
    std::string errorStr; // for any error msg, defaults to ""
    
    // validate network id
    if (!this->strToUint8(netIdStr, netId)) // unsuccessful conversion
      errorStr = std::string("Invalid network id ") + netIdStr +
      " - must be integer in [0..255]";
    
    else if (netMap.count(netId) != 0) // duplicate
      errorStr = std::string("Duplicate network id ") + netIdStr;
    
    else if (netMapAll.count(netId) == 0) // unknown id
      errorStr = std::string("Unknown network id ") + netIdStr;
    
    else  // all good in the neighborhood
      netMap.emplace(std::piecewise_construct,
                     std::forward_as_tuple(nnets++),
                     std::forward_as_tuple(std::move(netMapAll[netId])));

    if (errorStr != "") // log error if there is one
      this->logError(section, entry, errorStr, "ignoring");
  } // network id string loop
  
  if (netMap.size() == 0) // do we have any valid entries?
    this->logErrorFatal(section,
                        entry,
                        "No valid networks found - need one or more");
}

/**
 * Reads and returns the log level associated with the section name provided as an argument.
 * If log level is not found or invalid, provided default is used.
 *
 * @param section: name of configuration file section to read from.
 * @param defValue: default log level.
 * @return log level associated with provided section name, or default if invalid or not found.
 */
LogLevel WcConfigFile::logLevel(std::string const& section,
                                const LogLevel defValue) {
  
  LogLevel logLevel = defValue;
  
  const int logLevelInt = this->intValue(section, "log-level", defValue);
  
  if (logLevelInt >= 0 && logLevelInt < NLOG_LEVELS) /* is valid */
      logLevel = (LogLevel) logLevelInt;
  else this->logErrorDef(section, "log-level", "value out of range",
                   std::to_string(defValue), "0", std::to_string(NLOG_LEVELS));

  return logLevel;
}

/**
 * Reads and returns the IP port associated with the section name provided as an argument.
 * If port is not found or invalid, provided default is used.
 *
 * @param section: name of configuration file section to read from.
 * @param entry: name of entry within section to read port value from.
 * @param defValue: default port value.
 * @return IP port associated with provided section name, or default if invalid or not found.
 */
uint16_t WcConfigFile::port(std::string const& section,
                            std::string const& entry,
                            const uint16_t defValue) {

  uint16_t port = (uint16_t) this->intValue(section, entry, defValue);
  
  if (port < 1024 || port > 49151) {  // must be non-reserved & non-private
    this->logErrorDef(section, "log-level", "value out of range",
                   std::to_string(defValue), "1024", "49151");

    port = defValue;
  }
  
  return port;
}

/**
 * Returns the number of network defined in the section provided as an argument.
 *
 * @param section: name of configuration file section to read from.
 * @return the number of networks defined in section section.
 */
std::size_t WcConfigFile::nnetworks(std::string const& section) {

  NetworkMap netMap;
  this->networks(section, netMap); // delegate to standard reader

  return netMap.size();
}
