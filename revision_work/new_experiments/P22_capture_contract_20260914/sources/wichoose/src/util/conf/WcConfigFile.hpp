/**
 * Extends generic ConfigFile class with functionality specific to WiChoose programs.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#ifndef WC_CONFIG_FILE_HPP__
#define WC_CONFIG_FILE_HPP__

#include <cstdint> // uint*_t
#include <cstddef> // std::size_t

#include "../log/LogFile.hpp" // class LogLevel
#include "ConfigFile.hpp"     // class ConfigFile
#include "Network.hpp"        // type NetworkMap

class WcConfigFile : public ConfigFile {

public:
  /**
   * Creates a config object from the filename passed as argument.
   *
   * @param fname name of configuration file.
   */
  WcConfigFile(std::string const& fname);
    
  /**
   * Empty constructor creates a config object from the default filename.
   */
  WcConfigFile();
    
  /**
   * Reads and returns the log level associated with the section name provided as an argument.
   * If log level is not found or invalid, provided default is used.
   *
   * @param section: name of configuration file section to read from.
   * @param defValue: default log level.
   * @return log level associated with provided section name, or default if invalid or not found.
   */
  LogLevel logLevel(std::string const& section, const LogLevel defValue);

  /**
   * Reads and returns the IP port associated with the section name provided as an argument.
   * If port is not found or invalid, provided default is used.
   *
   * @param section: name of configuration file section to read from.
   * @param entry: name of entry within section to read port value from.
   * @param defValue: default port value.
   * @return IP port associated with provided section name, or default if invalid or not found.
   */
  uint16_t port(std::string const& section,
                std::string const& entry,
                const uint16_t defValue);
  
  /**
   * Reads network information from associated configuration file for section specified as an argument.
   * The networks are stored in the map that is passed in as an argument. The map is indexed by a unique
   * id that is local to the configuration section in question.
   *
   * @param section: name of configuration file section to read from.
   * @param netMap: the map onto which the read networks are to be saved.
   */
  void networks(std::string const& section, NetworkMap& netMap);
  
  /**
   * Returns the number of network defined in the section provided as an argument.
   *
   * @param section: name of configuration file section to read from.
   * @return the number of networks defined in section section.
   */
  std::size_t nnetworks(std::string const& section);

private:
  /**
   * Reads all network information from associated configuration file and saves them in the argument map.
   *
   * @param netMap: the map onto which the read networks are to be saved.
   */
  void networks(NetworkMap& netMap);
};

#endif // WC_CONFIG_FILE_HPP__
