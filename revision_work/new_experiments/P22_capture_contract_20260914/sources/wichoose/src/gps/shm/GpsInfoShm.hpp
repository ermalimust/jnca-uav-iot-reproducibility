/**
 * Defines a shared memory region to hold GPS information.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef GPS_INFO_SHM_HPP__
#define GPS_INFO_SHM_HPP__

#include "../../util/shm/Shm.hpp" // struct Shm

#include "../GpsInfo.hpp"   // struct GpsInfo

struct GpsInfoShm : Shm {
  GpsInfo gpsInfo;
};

#endif // GPS_INFO_SHM_HPP__
