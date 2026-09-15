/**
 * Implements class that provides a convenient way to read interface choice from the shared memory region.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <string>     // std::string
#include <sys/mman.h> // shm_open, mmap, etc
#include <pthread.h>  // pthread_mutex_lock, etc
#include <fcntl.h>    // O_RDWR, etc

#include "../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../util/log/LogFile.hpp"       // class LogFile

#include "WiChoiceReader.hpp" // class WiChoiceReader

#define WI_CHOICE_SHM_PATH_DEF "/wichoose-wichoice"

/**
 * Empty constructor.
 */
WiChoiceReader::WiChoiceReader() : ShmReader() { }

/**
 * Constructor that initializes the reader or not, depending on the boolean
 *
 * @param init: if true, the reader is initialized, otherwise it is not.
 */
WiChoiceReader::WiChoiceReader(const bool init) { if (init) this->init(); }

/**
 * Helper method that reads and loads configuration parameters (shm path and default interface)
 * from the config file.
 */
void WiChoiceReader::readConfig() {
  this->setShmPath("wichoice", WI_CHOICE_SHM_PATH_DEF);
}

/**
 * High-level helper that is meant to put the shared memory in a usable state, after the config has been
 * read.
 */
void WiChoiceReader::initShm() {
  
  // open and map shared memory
  this->wiChoiceShm = (WiChoiceShm *) this->attachShm(O_RDWR,
                                                      S_IRUSR | S_IRGRP,
                                                      sizeof(WiChoiceShm));
  
  this->initWiChoice(); // can't forget to initialize this->wiChoice
}

/**
 * Returns currently selected interface.
 *
 * @return currently selected interface.
 */
const std::string WiChoiceReader::getIfaceNow() {
  return this->getIface(false /*onUpdate*/);
}

/**
 * Blocks waiting for selected interface to be updated. As soon as it does, it returns it.
 *
 * @return interface that has just been updated.
 */
const std::string WiChoiceReader::getIfaceOnUpdate() {
  return this->getIface(true /*onUpdate*/);
}

/**
 * Helper for the public getter methods. It returns the iface name in the shared memory  either now or later
 * when there's an update, depending on the boolean argument.
 *
 * @param onUpdate: whether we should wait for an update before reading or not.
 */
const std::string WiChoiceReader::getIface(bool onUpdate){
  
  this->lockShm();

  if (onUpdate) waitForShmUpdate();

  const std::string iface(this->wiChoice->iface);

  this->unlockShm();

  return iface;
}

/**
 * Returns copy of current wifi choice.
 *
 * @return copy of current wifi choice.
 */
const WiChoice WiChoiceReader::getWiChoiceNow() {
  return this->getWiChoice(false /*onUpdate*/);
}

/**
 * Blocks waiting for wifi choice to be updated. As soon as it does, it returns a copy of it.
 *
 * @return copy of just-updated wifi choice.
 */
const WiChoice WiChoiceReader::getWiChoiceOnUpdate() {
  return this->getWiChoice(true /*onUpdate*/);
}

/**
 * Helper for the public getter methods. It returns the iface name in the shared memory  either now or later
 * when there's an update, depending on the boolean argument.
 *
 * @param onUpdate: whether we should wait for an update before reading or not.
 */
const WiChoice WiChoiceReader::getWiChoice(bool onUpdate){
  
  this->lockShm();

  if (onUpdate) waitForShmUpdate(); // wait if we should do so

  WiChoice wiChoice = *this->wiChoice; // copy over
  
  this->unlockShm();
  
  return wiChoice; // relies on named return value optimization (nvro) to
                   // prevent object copy in this situation
}
