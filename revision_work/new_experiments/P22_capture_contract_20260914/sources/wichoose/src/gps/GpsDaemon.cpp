/**
 * Abstract class GpsDaemon provides a template for the creation of background programs (daemons) that
 * periodically update GPS information that exists in a shared memory region.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <sstream> // std::stringstream

#include "../util/log/LogFile.hpp" // LOG_*
#include "shm/GpsInfoWriter.hpp"   // class GpsInfoWriter

#include "GpsDaemon.hpp"     // class GpsDaemon

/**
 * Empty constructor.
 */
GpsDaemon::GpsDaemon() : Runnable(), curGpsInfo() { }

/**
 * Empty destructor.
 */
GpsDaemon::~GpsDaemon() { }

/**
 * Returns a copy of the current gps info.
 *
 * @return a copy of the current gps info.
 */
const GpsInfo& GpsDaemon::getCurGpsInfo() const { return this->curGpsInfo; }

/**
 * Executes a loop of reading GPS information and copying it to the shared memory region, until
 * eventually the endProgram flag becomes true (thorugh some external intervention).
 *
 * Meant to be run as a thread.
 */
void GpsDaemon::run(){

  GpsInfoWriter gpsInfoWriter; // calls constructor, sets up shared memory

  LOG_MSG("GpsDaemon up and running");
  
  while (!this->endProgram.load()) { // main loop

    this->updateGpsInfo(); // read latest gps info

    gpsInfoWriter.setGpsInfo(this->curGpsInfo); // update shared memory

    this->printGpsInfo(); // print latests gps info
  }
    
  // note: gpsInfoWriter destructor will be called when it goes out of scope
}

/**
 * Prints current  gps information to the log.
 */
void GpsDaemon::printGpsInfo() const {

  std::stringstream ss;
  ss << "GPS systime=" << this->curGpsInfo.systime <<
    ", gpstime=" << this->curGpsInfo.gpstime <<
    ", lat=" << this->curGpsInfo.lat <<
    ", lon=" << this->curGpsInfo.lon <<
    ", alt=" << this->curGpsInfo.alt <<
    ", head=" << this->curGpsInfo.head <<
    ", speed=" << this->curGpsInfo.speed <<
    ", hdop=" << this->curGpsInfo.hdop <<
    ", nsats=" << (unsigned int) this->curGpsInfo.nsats <<
    ", fix=" << (unsigned int) this->curGpsInfo.fix;

  LOG_VERBOSE(ss.str().c_str());
}
