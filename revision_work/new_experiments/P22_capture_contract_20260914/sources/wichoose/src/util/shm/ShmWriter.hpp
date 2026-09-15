/**
 * Defines abstract class that acts as template for readers of shared memory regions..
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef SHM_WRITER_HPP__
#define SHM_WRITER_HPP__

#include "ShmReader.hpp" // ShmReader

class ShmWriter : virtual public ShmReader {
  
protected:
  /**
   * Empty constructor.
   */
  ShmWriter();

  /**
   * Destroys the object. First it wakes up any listeners, then it destroys the shared memory region.
   */
  ~ShmWriter();
  
  /**
   * Initializes the mutex and update condition variables that facilitate shared memory access.
   */
  void initMutexAndCond();

  /**
   * Destroys update condition and unlinks memory region.
   */
  void destroyShm();
  
};

#endif // SHM_WRITER_HPP__
