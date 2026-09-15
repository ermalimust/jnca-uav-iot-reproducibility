/**
 * Defines a class to compute the distance between the mobile node and the access point.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef AP_DIST_CALCULATOR_HPP__
#define AP_DIST_CALCULATOR_HPP__

#include "../../../gps/GpsInfo.hpp" // struct GpsInfo

class ApDistCalculator {

public:
  /**
   * Constructor method.
   */
  ApDistCalculator();

  /**
   * Computes and returns distance between ap and the node whose mobility information is provided as an
   * argument.
   *
   * @param gpsInfo: the node's mobility information.
   * @return calculated ap-node distance, in meters.
   */
  double apDist(GpsInfo const& gpsInfo) const;

  /**
   * Checks whether the argument mobility represents a point to the west of the access point.
   *
   * @return true if the argument location is west of the access point, and false otherwise.
   */
  bool isWestOfAp(GpsInfo const& gpsInfo) const;
  
private:
  // access point coordinates for relative position calculations
  double apLat;
  double apLon;
};

#endif // AP_DIST_CALCULATOR
