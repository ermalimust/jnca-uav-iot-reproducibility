/**
 * Defines abstract template for classes needing to protect access to some data with a mutex.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef LOCKABLE_HPP__
#define LOCKABLE_HPP__

#include <pthread.h> // pthread_mutex_t

class Lockable {

protected:
  pthread_mutex_t dataMutex; // pointer to shared memory-protecting mutex

  /**
   * Empty constructor.
   */
  Lockable();
  
  /**
   * Use mutex to lock access to data.
   */
  void lockData();

  /**
   * Use mutex to unlock access to data.
   */
  void unlockData();

};

#endif // LOCKABLE_HPP__
