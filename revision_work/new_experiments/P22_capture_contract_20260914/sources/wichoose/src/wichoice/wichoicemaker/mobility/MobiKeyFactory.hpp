/**
 * Defines factory to create MobiKey objects.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef MOBI_KEY_FACTORY_HPP__
#define MOBI_KEY_FACTORY_HPP__

#include "../../../gps/GpsInfo.hpp"            // struct GpsInfo
#include "../../../gps/shm/GpsInfoReader.hpp"  // class GpsInfoReader
#include "ApDistCalculator.hpp"                // class ApDistCalculator
#include "MobiKey.hpp"                         // class MobiKey

class MobiKeyFactory {

public:
  /**
   * Empty constructor.
   */
  MobiKeyFactory();
  
  /**
   * Creates and returns a MobiKey corresponding to the gps info passed in as an argument.
   *
   * @param gpsInfo: the gps information to gleam the mobility summary from.
   * @return the created MobiKey.
   */
  MobiKey create(GpsInfo const& gpsInfo);

  /**
   * Creates and returns a MobiKey corresponding to the node's current mobility.
   *
   * @return the created MobiKey.
   */
  MobiKey createCur();

private:
  GpsInfoReader gpsInfoReader;
  ApDistCalculator apDistCalculator;

  unsigned posRes;      // in meters
  unsigned dirRes;      // in degrees
  float highSpeedThres; // in Km/h
};

#endif // MOBI_KEY_FACTORY_HPP__
