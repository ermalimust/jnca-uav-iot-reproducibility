/**
 * Class featuring time-related utility functions.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <chrono>  // std::chrono
#include <time.h>  // clock_settime(), CLOCK_REALTIME, timespec,
#include <ctime>   // std::time_t, std::ctime()
#include <string>  // std::string
#include <sstream> // std::stringstream

#include "TimeUtil.hpp" // class TimeUtil

/**
 * Returns current system time, in milliseconds.
 *
 * @return current system time, in milliseconds.
 */
uint64_t TimeUtil::getSystimeMillis(){
  return std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().
                                    time_since_epoch()).
                                    count();
}

/**
 * Returns a string representing the current system date and time.
 *
 * @return string representing the current system date and time.
 */
std::string TimeUtil::getSystimeStr() {
  
  // retrieve current time
  const auto tpointNow = std::chrono::system_clock::now();
  const std::time_t timeNow = std::chrono::system_clock::to_time_t(tpointNow);
  
  // format it
  std::string dateTime(std::ctime(&timeNow));
  dateTime.erase(dateTime.find('\n')); // time final newline
  std::stringstream ss;
  ss << dateTime << " (unix " << timeNow << ")";
  
  return ss.str(); // return it
}


/**
 * Set the system time to that passed in as an argument.
 *
 * @param tstampSecs: time to set the system time to, in seconds from 12 AM, Jan 1, 1970.
 */
void TimeUtil::setSystime(uint32_t tstampSecs) {
  
  const struct timespec ts {(std::time_t) tstampSecs, 0}; // for clock_settime
  clock_settime(CLOCK_REALTIME, &ts); // set the realtime clock
}



