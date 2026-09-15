/**
 * Implements class that provides a convenient  way to write (and read) the selected interface from the shared
 * memory region.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <cstdint> // uint*_t
#include <cstring> // strncpy(), std::size_t
#include <string>  // std::string
#include <sstream> // std::stringstream
#include <fcntl.h> // O_RDWR, etc

#include "../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../util/log/LogFile.hpp"       // LOG_*

#include "WiChoiceWriter.hpp" // class WiChoiceWriter

#define WI_CHOICE_IFACE_NAME_DEF "wlan1"

/**
 * Empty constructor.
 */
WiChoiceWriter::WiChoiceWriter() : WiChoiceReader(), gpsInfoReader() { }

/**
 * Helper method that reads and loads configuration parameters (shm path and default interface)
 * from the config file.
 */
void WiChoiceWriter::readConfig() {

  // do whatever initialization the base class needs
  this->WiChoiceReader::readConfig();

  // read default interface name
  WcConfigFile configFile;
  this->ifaceDef = configFile.strValue("wichoice",
                                       "default-iface",
                                       WI_CHOICE_IFACE_NAME_DEF);

  // interface name can't be longer than what we support
  if (this->ifaceDef.length() > MAX_IFACE_NAME_LENGTH) {
    std::stringstream ss;
    ss << "Config exception: section=wichoice, value=default-iface, "
       << "invalid value " << this->ifaceDef
       << ". Length must be <= " << MAX_IFACE_NAME_LENGTH;
    LOG_FATAL_PERROR_EXIT(ss.str().c_str());
  }
}

/**
 * High-level helper that is meant to put the shared memory in a usable state, after the config has been
 * read.
 */
void WiChoiceWriter::initShm() {
  
  // open and map shared memory
  this->wiChoiceShm = (WiChoiceShm *) this->attachShm(O_CREAT | O_RDWR,
                                                      S_IRWXU | S_IRWXG,
                                                      sizeof(WiChoiceShm));
  
  this->initMutexAndCond(); // initialize mutex and update condition
    
  this->initWiChoice();     // can't forget to initialize this->wiChoice
  
  this->setIface(this->ifaceDef); // now all's set up, set default interface
}

/**
 * Sets shared memory iface to be equal to that passed in as an argument. If the interface is different
 * from the one already in the shared memory, listeners are notified that an update has taken place.
 *
 * @param iface: reference of the string to copy from. Its length must not be larger than
 *  MAX_IFACE_NAME_LENGTH, otherwise an error will be thrown.
 *
 *  @return true if the iface actually changed from the previous value, and false otherwise.
 */
bool WiChoiceWriter::setIface(std::string const& iface) {

  // make sure iface name is not too long
  const std::size_t ifaceLen = iface.length();
  
  if (ifaceLen > MAX_IFACE_NAME_LENGTH) {
    std::stringstream ss;
    ss << "Error trying to set interface to " << iface
       << ". Length must be <= " << MAX_IFACE_NAME_LENGTH;
    LOG_FATAL_PERROR_EXIT(ss.str().c_str());
  }
  
  bool retval = false; // assume no change as default
  
  this->lockShm(); // don't let anyone read while we're writing
  
  // check whether it is the same as what's already there
  if (iface != this->wiChoice->iface) { // copy iff different
        
    // copy iface name over
    strncpy(this->wiChoice->iface /*dst*/,
            iface.c_str() /*src*/,
            ifaceLen+1 /*len*/); // + 1 ensures '\0' termination
    
    // record the time at which this change occurred
    this->wiChoice->tstamp = this->gpsInfoReader.getGpstimeNow();

    retval = true; // means a change has occured
  }

  this->unlockShm(); // writing done, free up access

  if (retval) this->notifyListeners(); // inform listeners of change
  
  return retval;
}
