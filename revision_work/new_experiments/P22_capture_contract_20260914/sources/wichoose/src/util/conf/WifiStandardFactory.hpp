/**
 * Defines factory for creating WiFi standard enum values out of strings.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef WIFI_STANDARD_FACTORY_HPP__
#define WIFI_STANDARD_FACTORY_HPP__

#include <string> // std::string

#include "WifiStandard.hpp" // enum WifiStandard

class WifiStandardFactory {

public:
  /**
   * Returns a WifiStandard enum corresponding to the textual argument.
   *
   * @param wstandStr: textual representation of wifi standard.
   * @return a WifiStandard enum corresponding to the textual argument.
   * @throws std::invalid argument exception if argument does not represent supported standard.
   */
  static WifiStandard create(std::string const& wistandStr);
};

#endif // WIFI_STANDARD_FACTORY_HPP__
