/**
 * Defines abstract class that acts as template for readers of shared memory regions..
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef SHM_READER_HPP__
#define SHM_READER_HPP__

#include <sys/stat.h> // mode_t
#include <pthread.h>  // pthread_mutex_t
#include <cstddef>    // std::size_t
#include <string>     // std::string

#include "Shm.hpp" // struct Shm

class ShmReader {

public:
  /**
   * Tells us whether the shared memory has an active writer.
   *
   * @return true if it has an active writer, false it it does not.
   */
  bool isLive();

  /**
   * Broadcast change in state to shared memory region listeners, waking them up from their wait state.
   * Note: does not lock/unlock the region.
   */
  void notifyListeners();

protected:
  int shmfd;           // shared memory file descriptor
  Shm* shm;            // pointer to shared memory region
  std::string shmPath; // needed for unlink upon destruction
  std::size_t len;
  
  /**
   * Empty constructor.
   */
  ShmReader();
  
  /**
   * Destroys the object - closes the shared memory file descriptor.
   */
  ~ShmReader();
  
  /**
   * Helper method that sets the shm path from the config file.
   *
   * @param section: config file section to read from.
   * @param defValue: value to use if config file doesn't have one.
   */
  void setShmPath(std::string const& section, std::string const& defValue);

  /**
   * High-level helper that calls readConfig() and then initShm().
   * Every concrete-class constructor should call this.
   */
  void init();
  
  /**
   * Helper method that reads and loads configuration parameters from the config file.
   * Must be implemented by concrete subclasses.
   */
  virtual void readConfig() = 0;
  
  /**
   * High-level helper that is meant to put the shared memory in a usable state.
   * Must be implemented by concrete subclasses.
   */
  virtual void initShm() = 0;

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
  Shm* attachShm(int oflag, mode_t mode, std::size_t len);

  /**
   * Use mutex to lock access to the shared memory region.
   */
  void lockShm();

  /**
   * Use mutex to un lock access to the shared memory region.
   */
  void unlockShm();

  /**
   * Stops and waits for the condition associated with the shared memory to be updated before returning.
   * Note: assumes region mutex has been locked prior to call!
   */
  void waitForShmUpdate();
};

#endif // SHM_READER_HPP__
