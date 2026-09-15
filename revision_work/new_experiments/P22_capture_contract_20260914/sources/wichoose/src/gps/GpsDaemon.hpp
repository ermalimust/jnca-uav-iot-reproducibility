/**
 * Abstract class GpsDaemon provides a template for the creation of background programs (daemons) that
 * periodically update GPS information that exists in a shared memory region.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef GPS_DAEMON_HPP__
#define GPS_DAEMON_HPP__

#include "../util/thread/Runnable.hpp" // class Runnable
#include "GpsInfo.hpp"                 // struct GpsInfo

class GpsDaemon : public Runnable {

public:
  
  /**
   * Empty constructor.
   */
  GpsDaemon();
  
  /**
   * Empty destructor.
   */
  virtual ~GpsDaemon();
  
  /**
   * Executes a loop of reading GPS information and copying it to the shared memory region, until
   * eventually the endProgram flag becomes true (through some external intervention).
   *
   * Overrides Runnable::run().
   * Meant to be run as a thread.
   */
  void run() override;
  
  /**
   * Returns a copy of the current gps info.
   *
   * @return a copy of the current gps info.
   */
  const GpsInfo& getCurGpsInfo() const;

protected:
  GpsInfo curGpsInfo; // latest gps information that was read
  
  /**
   * Updates curGpsInfo field accord to the latest gps information from whatever source.
   * Must be overridden by concrete subclasses.
   */
  virtual void updateGpsInfo() = 0;

private:
  /**
   * Prints current  gps information to the log.
   */
  void printGpsInfo() const;

};

#endif // GPS_DAEMON_HPP__
