/**
 * Defines class that listens for changes in GPS information and prints them to the standard output.
 *
 * Rui Meireles  {@vassar.edu} 2021, 2024
 */

#ifndef WC_GPS_PRINTER_HPP__
#define WC_GPS_PRINTER_HPP__

#include "../../util/thread/Runnable.hpp" // class Runnable
#include "../shm/GpsInfoReader.hpp"       // class GpsInfoReader

class WcGpsPrinter : public Runnable {

public:
  
  /**
   * Constructor.
   *
   * @param nprints: the number of times to print the GPS information.
   */
  WcGpsPrinter(unsigned long long nprints);

  /**
   * Executes a loop of reading GPS information and printing it to the standard output, until we've printed
   * the requested number of times or the endProgram flag becomes true (through some external
   * intervention).
   *
   * Overrides Runnable::run().
   * Meant to be run as a thread.
   */
  void run() override;
  
  /**
   * Halts the thread. The superclass one isn't sufficient because the thread may be blocked waiting for
   * a condition to be updated. We must therefore trigger that update manually.
   */
  virtual void stop() override;

private:
  unsigned long long nprints;

  GpsInfoReader gpsInfoReader;
};

#endif // WC_GPS_PRINTER_HPP__
