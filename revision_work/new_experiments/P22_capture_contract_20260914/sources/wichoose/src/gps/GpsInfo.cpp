/**
 * Implements methods associated with GPS data structure GpsInfo.
 *
 * Rui Meireles  {@vassar.edu} 2025
 */

#include "GpsInfo.hpp" // struct GpsInfo

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
GpsInfo::GpsInfo(const uint64_t systime,
                 const uint32_t gpstime,
                 const uint8_t fix,
                 const uint8_t nsats,
                 const float hdop,
                 const float lat,
                 const float lon,
                 const float alt,
                 const float speed,
                 const float head) :
                 systime(systime), gpstime(gpstime), fix(fix), nsats(nsats),
                 hdop(hdop), lat(lat), lon(lon), alt(alt), speed(speed),
                 head(head) { }
