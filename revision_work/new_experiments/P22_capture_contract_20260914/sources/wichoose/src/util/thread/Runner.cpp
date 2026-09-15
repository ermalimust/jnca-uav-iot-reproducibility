/**
 * Convenience class for managing the lifecycle of Runnable objects.
 * It is capable of executing one or more of them, and stopping them upon kill signal reception.
 * Prevents code duplication that would occur if every Runnable had to be managed independently.
 *
 * Designed as a singleton due to the need for interacting with signal handlers.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <cstddef>  // std::size_t
#include <signal.h> // sigaction, SIGTERM, etc
#include <thread>   // std::thread
#include <sstream>  // std::stringstream
#include <memory>   // std::unique_ptr

#include "../log/LogFile.hpp" // LOG_*

#include "Runner.hpp"  // class Runner

/**
 * Returns singleton runner instance.
 *
 * @return reference to singleton instance.
 */
Runner& Runner::getInstance(){
  static Runner instance;
  return instance;
}

/**
 * Empty private constructor for singleton.
 */
Runner::Runner() : runnables() { }

/**
 * Add a Runnable object to the collection of objects to execute.
 *
 * @param runnable: reference to runnable object to add.
 */
void Runner::addRunnable(Runnable& runnable){
  this->runnables.push_back(&runnable);
}

/**
 * Remove and destroy all runnables
 */
void Runner::clearRunnables() { this->runnables.clear(); }

/**
 * Notifies the  runnables to stop executing.
 * Meant to be called upon the reception of a termination signal.
 * Must be static, which forces singleton design.
 */
void Runner::sigStopHandler(int){
  Runner& runner = Runner::getInstance();
  for (auto runnable : runner.runnables) runnable->stop();
}

/**
 * Launches all the runnable threads and then waits for them to end.
 */
void Runner::run() {
  // install signal handler
  struct sigaction sigbreak;
  sigbreak.sa_handler = &Runner::sigStopHandler;
  sigemptyset(&sigbreak.sa_mask);
  sigbreak.sa_flags = 0;
  
   if (sigaction(SIGINT, &sigbreak, NULL) != 0 ||
       sigaction(SIGTERM, &sigbreak, NULL) != 0 ||
       sigaction(SIGHUP, &sigbreak, NULL) != 0)
    LOG_FATAL_PERROR_EXIT("Error installing signal handler");

  // note how many runnables we have
  const std::size_t nrunnables = this->runnables.size();

  // iterate through the runnables and launch each one as a thread.
  // array for threads, must be on the heap because runtime-determined size
  std::unique_ptr<std::thread[]> threads(new std::thread[nrunnables]);
  unsigned int i = 0;
  for (auto runnable : this->runnables){
    threads[i++] = std::thread(&Runnable::run, runnable);
    
    // log the start
    std::stringstream ss;
    ss << typeid(*runnable).name() << " thread started";
    LOG_VERBOSE(ss.str().c_str());
  }
  
  // wait for all them threads to end
  i = 0;
  for (auto runnable : this->runnables){
    threads[i++].join();
    
    // log the join
    std::stringstream ss;
    ss << typeid(*runnable).name() << " thread joined";
    LOG_VERBOSE(ss.str().c_str());
  }
}
