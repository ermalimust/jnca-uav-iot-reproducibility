/**
 * Simple program that sets the device's realtime clock to the current GPS time.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include <iostream> // std::cout, std::cerr, std::endl

#include "../shm/GpsInfoReader.hpp"     // class GpsInfoReader
#include "../../util/time/TimeUtil.hpp" // class TimeUtil
#include "../../util/log/LogFile.hpp"   // LOG_*

#define LOG_FNAME "/var/log/wcgpsclocket.log"

/**
 * The procedure that actually bootstraps the program.
 * 
 * @param argc: number of command line arguments, including the program's name.
 * @param argv: array of command line arguments, as strings.
 * @return 0 on success, other value on error.
 */
int main(int argc, char *argv[]) {

  LOG_INIT(LOG_FNAME, "wcgpsclockset");

  LOG_VERBOSE("wcgpsclockset starting");
  
  if (argc > 1)
    std::cerr << "Ignoring all command-line arguments." << std::endl;

  // get gpstime as soon as it changes next
  GpsInfoReader gpsInfoReader;
  const uint32_t gpstime = gpsInfoReader.getGpstimeOnUpdate();

  TimeUtil::setSystime(gpstime); // set systime = gpstime

  // print out what we've done
  std::cout << "Systime is now: " << TimeUtil::getSystimeStr() << std::endl;

  LOG_CLOSE();

  return 0;
}
