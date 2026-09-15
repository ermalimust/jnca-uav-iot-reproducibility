/**
 * Implements abstract class that acts as template for writers of shared memory regions.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <pthread.h>  // pthread_*
#include <sys/mman.h> // shm_unlink

#include "../log/LogFile.hpp" // LOG_*

#include "ShmWriter.hpp" // class ShmWriter

/**
 * Empty constructor.
 */
ShmWriter::ShmWriter() { }

/**
 * Destroys the object. First it wakes up any listeners, then it destroys the shared memory region.
 */
ShmWriter::~ShmWriter() {
  this->destroyShm();
}

/**
 * Initializes the mutex and update condition variables that facilitate shared memory access.
 */
void ShmWriter::initMutexAndCond() {

  // initialize mutex
  pthread_mutexattr_t mattr;
  if (pthread_mutexattr_init(&mattr))
    LOG_FATAL_PERROR_EXIT("ShmWriter pthread_mutexattr_init()");
  
  if (pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED))
    LOG_FATAL_PERROR_EXIT("ShmWriter pthread_condattr_setpshared()");
    
  if (pthread_mutex_init(&this->shm->mutex, &mattr))
    LOG_FATAL_PERROR_EXIT("ShmWriter pthread_mutex_init()");
  
  // initialize info update condition variable
  pthread_condattr_t cattr;
  if (pthread_condattr_init(&cattr))
    LOG_FATAL_PERROR_EXIT("ShmWriter pthread_condattr_init()");

  if (pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED))
    LOG_FATAL_PERROR_EXIT("ShmWriter pthread_condattr_setpshared()");

  if (pthread_cond_init(&this->shm->updateCond, &cattr))
    LOG_FATAL_PERROR_EXIT("ShmWriter pthread_cond_init()");
  
  this->shm->isLive = true; // now live
}

/**
 * Destroys update condition and unlinks memory region.
 */
void ShmWriter::destroyShm() {
  
  this->shm->isLive = false; // no longer live
  
  if (pthread_cond_destroy(&this->shm->updateCond))
    LOG_FATAL_PERROR_EXIT("ShmWriter pthread_cond_destroy())");

  // bye bye gps shared memory region
  if (shm_unlink(this->shmPath.c_str()) != 0)
    LOG_FATAL_PERROR_EXIT("ShmWriter shm_unlink()");
}
