/**
 * Defines a class to compute the distance between the mobile node and the access point.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <string>  // std::string
#include <cmath>   // std::cos(), std::asin(), std::pow()

#include "../../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../../util/log/LogFile.hpp"       // LOG_*
 
#include "ApDistCalculator.hpp" // class ApDistCalculator

#define AP_LAT_DEF 41.177017 // default AP longitude, in decimal degrees
#define AP_LON_DEF -8.595887 // default AP longitude, in decimal degrees

/**
 * Constructor method.
 */
ApDistCalculator::ApDistCalculator() {
  
  WcConfigFile configFile;

  // read AP latitude and longitude
  const std::string section("wichoicemaker-mobility");
  this->apLat = configFile.doubleValue(section, "ap-lat", AP_LAT_DEF);
  this->apLon = configFile.doubleValue(section, "ap-lon", AP_LON_DEF);
}

/**
 * Helper method that computes the haversine distance, in meters, between the two points passed in
 * as arguments.
 * 
 * @param p1Lat: first point's latitude.
 * @param p1Lon: first point's longitude.
 * @param p2Lat: second point's latitude.
 * @param p2Lon: second point's longitude.
 * @return distance between the two points, in meters.
 */
double haversineDist(const double p1Lat, const double p1Lon,
                     const double p2Lat, const double p2Lon) {
  
  // distance between latitudes and longitudes
  const double dLat = (p2Lat - p1Lat) * M_PI / 180.0;
  const double dLon = (p2Lon - p1Lon) * M_PI / 180.0;
   
  // convert to radians
  const double p1LatRad = p1Lat * M_PI / 180.0;
  const double p2LatRad = p2Lat * M_PI / 180.0;
   
  // apply distance formula
  const double a = std::pow(std::sin(dLat / 2), 2) +
                   std::pow(std::sin(dLon / 2), 2) *
                   std::cos(p1LatRad) * std::cos(p2LatRad);
  const double c = 2 * std::asin(std::sqrt(a));
  
  const double erad = 6371000; // earth's radius, in meters
  
  return erad * c;
}

/**
 * Computes and returns distance between ap and the node whose mobility information is provided as an
 * argument.
 *
 * @param gpsInfo: the node's mobility information.
 * @return calculated ap-node distance, in meters.
 */
double ApDistCalculator::apDist(GpsInfo const& gpsInfo) const {
  return haversineDist(gpsInfo.lat, gpsInfo.lon, this->apLat, this->apLon);
}

/**
 * Checks whether the argument mobility represents a point to the west of the access point.
 *
 * @return true if the argument location is west of the access point, and false otherwise.
 */
bool ApDistCalculator::isWestOfAp(GpsInfo const& gpsInfo) const {
  return gpsInfo.lon < this->apLon;
}
