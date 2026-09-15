/**
 * Defines abstract template for classes needing to protect access to some data with a mutex.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <pthread.h>  // pthread_*

#include "../log/LogFile.hpp" // LOG_*

#include "Lockable.hpp" // class Lockable

/**
 * Empty constructor.
 */
Lockable::Lockable() {
  // init data protection mutex
  pthread_mutexattr_t mattr;
  if (pthread_mutexattr_init(&mattr))
    LOG_FATAL_PERROR_EXIT("Lockable pthread_mutexattr_init()");

  if (pthread_mutex_init(&this->dataMutex, &mattr))
    LOG_FATAL_PERROR_EXIT("Lockable pthread_mutex_init()");
}

/**
 * Use mutex to lock access to data.
 */
void Lockable::lockData() {
  if (pthread_mutex_lock(&this->dataMutex))
    LOG_FATAL_PERROR_EXIT("Lockable pthread_mutex_lock()");
}

/**
 * Use mutex to unlock access to data.
 */
void Lockable::unlockData() {
  if (pthread_mutex_unlock(&this->dataMutex))
    LOG_FATAL_PERROR_EXIT("Lockable pthread_mutex_unlock()");
}
