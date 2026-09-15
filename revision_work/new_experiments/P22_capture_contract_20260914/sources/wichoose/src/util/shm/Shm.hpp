/**
 * Defines a basic data structure for a lockable shared memory, featuring a mutex and an update condition.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef SHM_HPP__
#define SHM_HPP__

#include <pthread.h> // pthread_*

struct Shm {
  bool isLive = false;       // tells us whether the shm has an active writer
  pthread_mutex_t mutex;     // exclusive access
  pthread_cond_t updateCond; // supports update notification
};

#endif // SHM_HPP__
