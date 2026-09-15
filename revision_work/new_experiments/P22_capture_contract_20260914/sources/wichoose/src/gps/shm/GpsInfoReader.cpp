/**
 * Implements class that provides a convenient way to read GPS information from the shared memory region.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <fcntl.h> // O_RDWR, etc
#include <time.h>  // tzset()
#include <cstdlib> // setenv()
#include <string>  // std::string

#include "../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../util/log/LogFile.hpp"       // LOG_*
#include "GpsInfoShm.hpp"                   // struct GpsInfoShm

#include "GpsInfoReader.hpp" // class GpsInfoReader

#define GPS_INFO_SHM_PATH_DEF "/wichoose-gpsinfo"
#define GPS_INFO_TIMEZONE_DEF "Europe/Lisbon"

/**
 * Empty constructor. It always initializes the reader.
 */
GpsInfoReader::GpsInfoReader() : GpsInfoReader(true) { }

/**
 * Constructor that initializes the reader or not, depending on the boolean
 *
 * @param init: if true, the reader is initialized, otherwise it is not.
 */
GpsInfoReader::GpsInfoReader(const bool init) { if (init) this->init(); }

/**
 * Helper method that reads and loads configuration parameters (shm path and timezone)
 * from the config file.
 */
void GpsInfoReader::readConfig() {
  
  this->setShmPath("gps-info", GPS_INFO_SHM_PATH_DEF);

  WcConfigFile configFile;
  
  // read and set timezone
  std::string timezone = configFile.strValue("gps-info",
                                             "timezone",
                                             GPS_INFO_TIMEZONE_DEF);
  setenv("TZ", timezone.c_str(), 1 /*overwrite*/);
  tzset();
}

/**
 * High-level helper that is meant to put the shared memory in a usable state, after the config has been
 * read.
 */
void GpsInfoReader::initShm() {
  // open and map shared memory
  this->gpsInfoShm = (GpsInfoShm *) this->attachShm(O_RDWR,
                                                    S_IRUSR | S_IRGRP,
                                                    sizeof(GpsInfoShm));
}

/**
 * Copies GPS information from the shared memory to the structure passed in as an argument, right now.
 *
 * @param gpsInfo: reference of the the gps info struct to copy to.
 */
void GpsInfoReader::getGpsInfoNow(GpsInfo& gpsInfo) {
  this->getGpsInfo(gpsInfo, false /*onUpdate*/);
}

/**
 * Copies GPS information from the shared memory to the structure passed in as an argument,
 * but only when the data is updated by the GPS daemon. It blocks waiting until then.
 *
 * @param gpsInfo: reference of the the gps info struct to copy to.
 */
void GpsInfoReader::getGpsInfoOnUpdate(GpsInfo& gpsInfo) {
  this->getGpsInfo(gpsInfo, true /*onUpdate*/);
}


/**
 * Helper for the public getter methods. It  copies GPS information from the shared memory to the
 * structure passed in as an argument either now or later when there's an update, depending on the
 * boolean argument.
 *
 * @param gpsInfo: reference of the the gps info struct to copy to.
 * @param onUpdate: whether we should wait for an update before copying or not.
 */
void GpsInfoReader::getGpsInfo(GpsInfo& gpsInfo, bool onUpdate) {
  
  this->lockShm();
  if (onUpdate) this->waitForShmUpdate();

  // do the actual copying
  gpsInfo = this->gpsInfoShm->gpsInfo; // shallow copy is sufficient

  this->unlockShm();
}

/**
 * Retrieve and return the current GPS time.
 *
 * @return current GPS time.
 */
uint32_t GpsInfoReader::getGpstimeNow() const {
  return this->gpsInfoShm->gpsInfo.gpstime; // no locking because single value
}

/**
 * Block waiting for the GPS time to be updated, then return its new value.
 *
 * @return recently updated GPS time.
 */
uint32_t GpsInfoReader::getGpstimeOnUpdate() {
  
  this->lockShm();
  this->waitForShmUpdate();

  const uint32_t gpstime = this->gpsInfoShm->gpsInfo.gpstime;

  this->unlockShm();

  return gpstime;
}
