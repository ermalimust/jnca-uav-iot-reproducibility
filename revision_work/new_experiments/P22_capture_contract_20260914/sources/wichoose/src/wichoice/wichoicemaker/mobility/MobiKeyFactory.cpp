/**
 * Implements factory that creates throughput estimators.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include <cstdint> // uint*_t

#include "../../../gps/GpsInfo.hpp"            // struct GpsInfo
#include "../../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "MobiKey.hpp"                         // class MobiKey

#include "MobiKeyFactory.hpp" // class MobiKeyFactory

#define POS_RES_DEF 10  // meters
#define POS_RES_MIN 1
#define POS_RES_MAX 100

#define DIR_RES_DEF 180 // degrees
#define DIR_RES_MIN 1
#define DIR_RES_MAX 360

#define HIGH_SPEED_THRES_DEF 10 // Km/h
#define HIGH_SPEED_THRES_MIN 0
#define HIGH_SPEED_THRES_MAX 200

/**
 * Empty constructor.
 */
MobiKeyFactory::MobiKeyFactory() : gpsInfoReader(), apDistCalculator() {
  
  // read the configuration
  WcConfigFile configFile;
  const std::string section = "wichoicemaker-mobility";
  
  this->posRes = configFile.uintValue(section, 
                                      "pos-res",
                                      POS_RES_DEF,
                                      POS_RES_MIN, 
                                      POS_RES_MAX);
  

  this->dirRes = configFile.uintValue(section, 
                                      "dir-res",
                                      DIR_RES_DEF,
                                      DIR_RES_MIN,
                                      DIR_RES_MAX);
  
  this->highSpeedThres = configFile.floatValue(section,
                                               "high-speed-thres",
                                               HIGH_SPEED_THRES_DEF,
                                               HIGH_SPEED_THRES_MIN,
                                               HIGH_SPEED_THRES_MAX);
}

/**
 * Creates and returns a MobiKey corresponding to the gps info passed in as an argument.
 *
 * @param gpsInfo: the gps information to gleam the mobility summary from.
 * @return the created MobiKey.
 */
MobiKey MobiKeyFactory::create(GpsInfo const& gpsInfo) {

  // set relative position from AP (integer division)
  int rpos = ((unsigned long) apDistCalculator.apDist(gpsInfo) /
               this->posRes) * this->posRes;

  // if the client is west of the ap, make the position negative
  if (this->apDistCalculator.isWestOfAp(gpsInfo)) rpos *= -1;

  // set direction (integer division)
  const uint16_t dir = ((unsigned) gpsInfo.head) / this->dirRes * this->dirRes;

  // client moving at speed or not?
  const bool hispeed = gpsInfo.speed >= this->highSpeedThres;
  
  return MobiKey(rpos, dir, hispeed);
}

/**
 * Creates and returns a MobiKey corresponding to the node's current mobility.
 * @return the created MobiKey.
 */
MobiKey MobiKeyFactory::createCur() {

  GpsInfo gpsInfo;
  this->gpsInfoReader.getGpsInfoNow(gpsInfo);

  return this->create(gpsInfo);
}
