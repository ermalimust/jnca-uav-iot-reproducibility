/**
 * Defines a data structure to hold GPS information.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#ifndef GPS_INFO_HPP__
#define GPS_INFO_HPP__

#include <cstdint> // uint*_t

struct GpsInfo {
  uint64_t systime; // in millis
  uint32_t gpstime; // in seconds

  uint8_t fix;   // 1=nofix, 2=2D fix, 3=3D fix
  uint8_t nsats; // number of visible satellites
  float hdop;    // horizontal dilution of precision (lower is better)
	
	float lat; // decimal degrees
	float lon; // decimal degrees
	
	float alt;   // meters
	float speed; // Km/h
	float head;  // degrees from north

  /**
   * Constructor method.
   *
   * @param systime: system time, in milliseconds. Optional, defaults to 0.
   * @param gpsime: gps time, in seconds. Optional, defaults to 0.
   * @param fix: gps fix (1=nofix, 2=2D fix, 3=3D fix). Optional, defaults to 1.
   * @param nsats: number of visible satellites. Optional, defaults to 0.
   * @param hdop: horizontal dilution of precision. Optional, defaults to 0.
   * @param lat: latitude, in decimal degrees. Optional, defaults to 0.
   * @param lon: longitude, in decimal degrees. Optional, defaults to 0.
   * @param alt: altitude, in meters Optional, defaults to 0.
   * @param speed: speed, in Km/h. Optional, defaults to 0.
   * @param head: heading, in degrees from north. Optional, defaults to 0.
   */
  GpsInfo(const uint64_t systime = 0,
          const uint32_t gpstime = 0,
          const uint8_t fix = 1,
          const uint8_t nsats = 0,
          const float hdop = 0,
          const float lat = 0,
          const float lon = 0,
          const float alt = 0,
          const float speed = 0,
          const float head = 0);
};

#endif // GPS_INFO_HPP__
