/**
 * Implements class that periodically print each interface's RSSI.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include <iostream> // std::cout, std::cerr, std::endl
#include <string>   // std::string

#include "../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../util/conf/Network.hpp"      // type NetworkId
#include "../../util/thread/Runner.hpp"     // class Runner
#include "../../util/log/LogFile.hpp"       // LOG_*
#include "../../gps/shm/GpsInfoReader.hpp"  // class GpsInfoReader
#include "../chaninfo/ChanInfo.hpp"         // type ChannInfo, RSSI_DISCONNECTED
#include "../chaninfo/ChanInfoReader.hpp"   // class ChannInfoReader

#include "WcRssiPrinter.hpp" // class WiRssiPrinter

#define LOG_FNAME "/var/log/wcrssiprinter.log"

/**
 * Empty constructor.
 */
WcRssiPrinter::WcRssiPrinter() : netMap() {
  
  // read networks
  WcConfigFile configFile;
  configFile.networks("data-transfer", this->netMap);
}

/**
 * Executes a loop of determining and printing RSSI for each interface until the endProgram flag becomes
 * true (through some external intervention).
 *
 * Overrides Runnable::run().
 * Meant to be run as a thread.
 */
void WcRssiPrinter::run() {

  GpsInfoReader gpsInfoReader; // to trigger updates
  GpsInfo gpsInfo;
  
  // to read channel information
  ChanInfoReader chanInfoReader;
  ChanInfoMap const& chanInfoMap = chanInfoReader.getChanInfoMap();
  
  // print header line with format
  std::cout << "gpstime, iface, rssi" << std::endl;

  while (!this->endProgram.load()) { // main loop

    gpsInfoReader.getGpsInfoOnUpdate(gpsInfo); // blocks until update

    // if we were awaken due to the program ending we want to stop immediately
    if (this->endProgram.load()) break;

    chanInfoReader.updateChanInfo(); // update channel information

    for (auto& kvp : this->netMap) { // for each network
      const NetworkId netId = kvp.first;
      Network const& net = kvp.second;
  
      // do we have rssi info for it?
      auto chitr = chanInfoMap.find(netId);
      const int rssi = chitr != chanInfoMap.end() ? chitr->second.rssi :
                                                    RSSI_DISCONNECTED;

      // print the info out
      std::cout << gpsInfo.gpstime << ", "
                << net.iface << ", "
                << rssi << std::endl;
    } // network loop end
  } // main loop end
}

// WcRssiPrinter class implementation end

/**
 * The procedure that actually bootstraps the program.
 */
int main(int argc, char *argv[]) {

  LOG_INIT(LOG_FNAME, "wcrssiprinter");

  LOG_MSG("wcrssiprinter starting");
  
  if (argc > 1)
    std::cerr << "Ignoring all command-line arguments." << std::endl;

  WcRssiPrinter wcRssiPrinter; // create runnable

  Runner& runner = Runner::getInstance(); // get runner
  runner.addRunnable(wcRssiPrinter);      // add runnable to runner
  runner.run();                           // actually run the runnable

  LOG_CLOSE();

  return 0;
}
