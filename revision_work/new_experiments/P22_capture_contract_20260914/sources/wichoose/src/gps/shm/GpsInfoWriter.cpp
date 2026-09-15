/**
 * Defines class that provides a convenient  way to write (and read) GPS information from the shared
 * memory region.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <fcntl.h> // O_RDWR, etc

#include "GpsInfoShm.hpp" // struct GpsInfoShm

#include "GpsInfoWriter.hpp" // class GpsInfoWriter

/**
 * Empty constructor.
 */
GpsInfoWriter::GpsInfoWriter() : GpsInfoReader(false) { this->init(); }

/**
 * High-level helper that is meant to put the shared memory in a usable state, after the config has been
 * read.
 */
void GpsInfoWriter::initShm() {
  
  // open and map shared memory
  this->gpsInfoShm = (GpsInfoShm *) this->attachShm(O_CREAT | O_RDWR,
                                                    S_IRWXU | S_IRWXG,
                                                    sizeof(GpsInfoShm));
  
  this->initMutexAndCond(); // initialize mutex and update condition
}

/**
 * Copies GPS information from the struct passed in as argument to shared memory region.
 *
 * @param gpsInfo: reference of the the gps info struct to copy from.
 */
void GpsInfoWriter::setGpsInfo(GpsInfo const& gpsInfo) {

  this->lockShm();
  
  // limit update rate to 1 Hz
  if (this->gpsInfoShm->gpsInfo.gpstime != gpsInfo.gpstime) {
    
    this->gpsInfoShm->gpsInfo = gpsInfo; // shallow copy good because no nesting
    
    this->notifyListeners(); // new gps info available
  }

  this->unlockShm();
}
