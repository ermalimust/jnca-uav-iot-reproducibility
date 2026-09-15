/**
 * Implements factory that creates WiFi standard enum values out of strings.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <map>       // std::map
#include <stdexcept> // std::invalid_argument

#include "WifiStandardFactory.hpp" // enum WifiStandard

/**
 * Returns a WifiStandard enum corresponding to the textual argument.
 *
 * @param wifiStandard: textual representation of wifi standard.
 * @return a WifiStandard enum corresponding to the textual argument.
 * @throws std::invalid argument exception if argument does not represent supported standard.
 */
WifiStandard WifiStandardFactory::create(std::string const& wiStandStr) {

  // static map with all the options
  static const std::map<std::string,WifiStandard> strToWiStandMap =
                                                    {{"n", WifiStandard::N},
                                                     {"ac", WifiStandard::AC},
                                                     {"ad", WifiStandard::AD}};
  
  auto itr = strToWiStandMap.find(wiStandStr); // is provided string in map?
  
  if (itr == strToWiStandMap.end())
    throw std::invalid_argument(
     "Invalid wifi standard string. Only \"n\", \"ac\", and \"ad\" supported.");
    
  return itr->second; // the value associated with the key
}
