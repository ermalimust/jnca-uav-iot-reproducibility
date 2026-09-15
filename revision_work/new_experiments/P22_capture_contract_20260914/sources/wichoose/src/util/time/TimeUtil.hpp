/**
 * Class featuring time-related utility functions.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef TIME_UTIL_HPP__
#define TIME_UTIL_HPP__

#include <cstdint> // uint*_t
#include <string>  // std::string

class TimeUtil {

public:
  /**
   * Returns current system time, in milliseconds.
   *
   * @return current system time, in milliseconds.
   */
  static uint64_t getSystimeMillis();
  
  /**
   * Returns a string representing the current system date and time.
   *
   * @return string representing the current system date and time.
   */
  static std::string getSystimeStr();

  /**
   * Set the system time to that passed in as an argument.
   *
   * @param tstampSecs: time to set the system time to, in seconds from 12 AM, Jan 1, 1970.
   */
  static void setSystime(uint32_t tstampSecs);

};

#endif // TIME_UTIL_HPP__
