/**
 * Defines class to periodically print each interface's RSSI.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#ifndef WC_RSSI_PRINTER_HPP__
#define WC_RSSI_PRINTER_HPP__

#include "../../util/thread/Runnable.hpp" // class Runnable
#include "../../util/conf/Network.hpp"    // type NetworkMap

class WcRssiPrinter : public Runnable {

public:
  /**
   * Empty constructor.
   */
  WcRssiPrinter();

  /**
   * Executes a loop of determining and printing RSSI for each interface until the endProgram flag becomes
   * true (through some external intervention).
   *
   * Overrides Runnable::run().
   * Meant to be run as a thread.
   */
  void run() override;
  
private:  
  NetworkMap netMap; // map with all relevant
};

#endif // WC_RSSI_PRINTER_HPP__
