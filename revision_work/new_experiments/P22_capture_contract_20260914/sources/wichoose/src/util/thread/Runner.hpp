/**
 * Convenience class for managing the lifecycle of Runnable objects.
 * It is capable of executing one or more of them, and stopping them upon kill signal reception.
 * Prevents code duplication that would occur if every Runnable had to be managed independently.
 * 
 * Designed as a singleton due to the need for interacting with signal handlers.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef RUNNER_HPP__
#define RUNNER_HPP__

#include <vector> // std::vector

#include "Runnable.hpp"  // class Runnable

class Runner {

public:
  /**
   * Returns singleton runner instance.
   *
   * @return reference to singleton instance.
   */
  static Runner& getInstance();
  
  /**
   * Add a Runnable object to the collection of objects to execute.
   *
   * @param runnable: reference to runnable object to add.
   */
  void addRunnable(Runnable& runnable);

  /**
   * Remove and destroy all runnables
   */
  void clearRunnables();
  
  /**
   * Launches all the runnable threads and then waits for them to end.
   */
  void run();

  /**
   * Notifies the runnables to stop executing.
   * Meant to be called upon the reception of a termination signal.
   * Must be static, which forces singleton design.
   */
  static void sigStopHandler(int);


private:
  std::vector<Runnable*> runnables; // collection of runnables to execute
  
  /**
   * Empty private constructor for singleton.
   */
  Runner();
};

#endif // RUNNER_HPP__
