/**
 * Defines class for a GPS daemon that uses fake, preset, data.
 * The node never moves. It stays fixed at the location indicated in the configuration file.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <chrono>    // std::chrono
#include <thread>    // std::this_thread
#include <iostream>  // std::cout, std::endl
#include <iomanip>   // std::setprecision
#include <sstream>   // std::stringstream, std::endl

#include "../../util/conf/WcConfigFile.hpp" // class WcConfigFile

#include "WcGpsdFakerFix.hpp" // class WcGpsdFaker

#define FIX_DEF 1
#define NSATS_DEF 0
#define HDOP_DEF 99.0

#define LAT_DEF 40.0
#define LON_DEF -8.0

#define ALT_DEF 0.0
#define SPEED_DEF 0.0
#define HEAD_DEF 0.0

/**
 * Empty constructor.
 */
WcGpsdFakerFix::WcGpsdFakerFix() : GpsDaemon() {
  
  WcConfigFile configFile; // calls default constructor

  // read all other parameters
  this->curGpsInfo.fix = configFile.ubyteValue("wcgpsdfaker", 
                                               "fix",
                                               FIX_DEF);
  
  this->curGpsInfo.nsats = configFile.ubyteValue("wcgpsdfaker", 
                                                 "nsats",
                                                 NSATS_DEF);
  this->curGpsInfo.hdop = configFile.floatValue("wcgpsdfaker", 
                                                "hdop",
                                                HDOP_DEF);

  this->curGpsInfo.lat = configFile.floatValue("wcgpsdfaker", "lat", LAT_DEF);
  this->curGpsInfo.lon = configFile.floatValue("wcgpsdfaker", "lon", LON_DEF);

  this->curGpsInfo.alt = configFile.floatValue("wcgpsdfaker", "alt", ALT_DEF);
  this->curGpsInfo.speed = configFile.floatValue("wcgpsdfaker", 
                                                 "speed",
                                                 SPEED_DEF);
  this->curGpsInfo.head = configFile.floatValue("wcgpsdfaker", 
                                                "head",
                                                HEAD_DEF);
}

/**
 * Empty destructor.
 */
WcGpsdFakerFix::~WcGpsdFakerFix() { }

/**
 * Updates curGpsInfo field according to the latest systime. Everything else is constant.
 * Overrides same-name method from superclass.
 */
void WcGpsdFakerFix::updateGpsInfo() {

  uint32_t systimeSecs = this->sleepUntilNextSec(); // wait for systime change
  
  // save time info
  this->curGpsInfo.gpstime = systimeSecs;
  this->curGpsInfo.systime = systimeSecs*1000LLU; // LLU to make it 64 bit
  
  const std::streamsize ogprec = std::cout.precision(); // original precision
  std::stringstream ss;
  ss << "WcGpsdFakerMobi setting gpstime=" << this->curGpsInfo.gpstime <<
        ", systime=" << this->curGpsInfo.systime <<
        std::setprecision(10) << // increase precision for lat and lon
        ", lat=" << this->curGpsInfo.lat <<
        ", lon=" << this->curGpsInfo.lon <<
        std::setprecision(ogprec) << // restore default precision
        ", alt=" << this->curGpsInfo.alt <<
        ", speed=" << this->curGpsInfo.speed <<
        ", head=" << this->curGpsInfo.head;
  LOG_VERBOSE(ss.str().c_str());
}

/**
 * Helper method that pauses execution until the system time changes to the next second.
 *
 * @return the timestamp after waking up from sleep, in seconds.
 */
uint32_t WcGpsdFakerFix::sleepUntilNextSec() {
  
  using std::chrono::system_clock; // enable shorthand
  
  // determine current time
  const auto nowTimePoint = system_clock::now().time_since_epoch();
  const auto nextSecs = std::chrono::duration_cast
                        <std::chrono::seconds>(nowTimePoint).count() + 1;

  // sleep until next second
  std::this_thread::sleep_until(system_clock::from_time_t(nextSecs));

  return nextSecs;
}
