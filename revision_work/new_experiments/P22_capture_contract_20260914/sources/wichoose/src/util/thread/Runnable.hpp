/**
 * Abstract class for objects that can be run and stopped.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef RUNNABLE_HPP__
#define RUNNABLE_HPP__

#include <atomic> // std::atomic

class Runnable {

public:
  
  /**
   * Empty constructor. Initializes endProgram to false.
   */
  Runnable();
  
  /**
   * Starts executing whatever task the runnable is responsible for.
   * Meant to be executed as a thread.
   */
  virtual void run() = 0;
  
  /**
   * Signals the work thread to stop executing.
   * Can be overidden, but a default implementation that simply sets endProgram to true  is provided.
   */
  virtual void stop();
  
protected:
  std::atomic<bool> endProgram; // flag set to true when it's time to stop

};

#endif // RUNNABLE_HPP__
