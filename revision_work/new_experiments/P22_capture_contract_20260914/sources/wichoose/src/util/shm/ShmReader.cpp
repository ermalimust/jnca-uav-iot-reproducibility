/**
 * Implements abstract class that acts as template for readers of shared memory regions.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <sys/mman.h> // shm_open, mmap, etc
#include <pthread.h>  // pthread_*
#include <fcntl.h>    // O_RDWR, etc
#include <unistd.h>   // ftruncate(), close()
#include <sstream>    // std::stringstream

#include "../conf/WcConfigFile.hpp" // class WcConfigFile
#include "../log/LogFile.hpp"       // LOG_*

#include "ShmReader.hpp" // class ShmReader

/**
 * Empty constructor.
 */
ShmReader::ShmReader() : shmfd(0) { }


/**
 * Destroys the object - closes the shared memory file descriptor.
 */
ShmReader::~ShmReader() {
  if (close(this->shmfd) < 0) LOG_FATAL_PERROR("~ShmReader() close(shmfd)");
}

/**
 * Helper method that sets the shm path from the config file.
 *
 * @param section: config file section to read from.
 * @param defValue: value to use if config file doesn't have one.
 */
void ShmReader::setShmPath(std::string const& section,
                           std::string const& defValue) {

  WcConfigFile configFile;
  this->shmPath = configFile.strValue(section, "shm-path", defValue);
}

/**
 * High-level helper that calls readConfig() and then initShm().
 * Every concrete-class constructor should call this.
 */
void ShmReader::init() {
  this->readConfig();
  this->initShm();
}

/**
 * Helper method that opens and maps the shared memory region referenced by shmPath.
 * As a side effect it sets the  shared memory region pointer associated with the object.  Also returns it,
 * for convenience.
 *
 * @param oflag: the open flags to be passed in to the shm_open() call.
 * @param mode: the mode to be passed in to the shm_open() call.
 * @param len: size of memory region, in bytes.
 *
 * @return a pointer to the attached memory region.
 */
Shm* ShmReader::attachShm(int oflag, mode_t mode, std::size_t len) {

  this->len = len;
  // open the shared memory
  if ((this->shmfd = shm_open(this->shmPath.c_str(), oflag, mode)) < 0) {
    std::stringstream ss;
    ss << "ShmReader shm_open(" << this->shmPath.c_str() << ")";
    LOG_FATAL_PERROR_EXIT(ss.str().c_str());
  }
  
  // adjust shared memory segment to desired size (otherwise bus error!)
  ftruncate(this->shmfd, len);
  
  // request the shared segment
  if ((this->shm = (Shm*) mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED,
                               this->shmfd, 0)) == MAP_FAILED)
    LOG_FATAL_PERROR_EXIT("ShmReader mmap()");
  
  return this->shm;
}

/**
 * Use mutex to lock access to the shared memory region.
 */
void ShmReader::lockShm() {
  if (pthread_mutex_lock(&this->shm->mutex))
    LOG_FATAL_PERROR_EXIT("ShmReader pthread_mutex_lock()");
}

/**
 * Use mutex to un lock access to the shared memory region.
 */
void ShmReader::unlockShm() {
  if (pthread_mutex_unlock(&this->shm->mutex))
    LOG_FATAL_PERROR_EXIT("ShmReader pthread_mutex_unlock()");
}

/**
 * Stops and waits for the condition associated with the shared memory to be updated before returning.
 * Note: assumes region mutex has been locked prior to call!
 */
void ShmReader::waitForShmUpdate() {
  /* note on behavior: call to cond_wait unlocks mutex then blocks
     on condition. when condition is signaled, mutex is auto-locked. */
  if (pthread_cond_wait(&this->shm->updateCond, &this->shm->mutex))
    LOG_FATAL_PERROR_EXIT("ShmReader pthread_cond_wait()");
}

/**
 * Tells us whether the shared memory has an active writer.
 *
 * @return true if it has an active writer, false it it does not.
 */
bool ShmReader::isLive() { return this->shm->isLive; }

/**
 * Broadcast change in state to shared memory region listeners, waking them up from their wait state.
 * Note: does not lock/unlock the region.
 */
void ShmReader::notifyListeners() {
  if (pthread_cond_broadcast(&this->shm->updateCond))
    LOG_FATAL_PERROR_EXIT("ShmReader pthread_cond_broadcast()");
}
