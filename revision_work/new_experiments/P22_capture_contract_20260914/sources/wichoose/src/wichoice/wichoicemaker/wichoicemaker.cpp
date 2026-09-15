/**
 * Implements a class that periodically chooses the best network to use as a function of present mobility and
 * past mobility-indexed network performance.
 * It is an abstract class. Concrete subclasses must decide whether to use estimated or measurement
 * data in their computations.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <thread>   // std::this_thread
#include <sstream>  // std::stringstream
#include <cstdint>  // std::uint*_t
#include <cassert>  // assert()
#include <cstddef>  // std::size_t
#include <utility>  // std::move
#include <iostream> // std::cout, std::cerr, std::endl
#include <memory>   // std::unique_ptr

#include "../../util/thread/Runner.hpp"      // class Runner
#include "../../util/log/LogFile.hpp"        // LOG_*
#include "../../util/conf/WcConfigFile.hpp"  // class WcConfigFile
#include "../../gps/shm/GpsInfoReader.hpp"   // class GpsInfoReader
#include "../shm/WiChoiceWriter.hpp"         // class WiChoiceWriter
#include "database/NperfMemDb.hpp"           // class NperfMemDb
#include "measurement/FbackReceiver.hpp"     // class FbackReceiver
#include "estimation/TputEstAggr.hpp"        // class TputEstAggr
#include "nperfsum/NperfSum.hpp"             // struct NperfSums
#include "nperfsum/NperfSumStrat.hpp"        // class NperfSumStrat
#include "nperfsum/NperfSumStratFactory.hpp" // class NperfSumStratFactory
#include "WiChoiceMakerMes.hpp"              // class WiChoiceMakerMest
#include "WiChoiceMakerEst.hpp"              // class WiChoiceMakerEst

#include "WiChoiceMaker.hpp" // class WiChoiceMaker

#define LOG_FNAME "/var/log/wichoicemaker.log"

#define NET_CHOICE_DEF 0      // default network choice
#define NET_SWITCH_TIME_DEF 1 // default network switching time (s)

/**
 * Constructor method.
 *
 * @param nperfDb: historical network performance database to use when picking an interface.
 * @param nperfSumStrat: network performance summarization strategy to use.
 */
WiChoiceMaker::WiChoiceMaker(NperfDb& nperfDb,
                             NperfSumStrat const& nperfSumStrat,
                             std::string const& printTag) :
                                              nperfSumStrat(nperfSumStrat),
                                              nperfDb(nperfDb),
                                              printTag(printTag),
                                              netMap() {

  // configure parameters from config file
  WcConfigFile configFile;

  // read networks
  configFile.networks("wichoice", this->netMap);
  this->nnets = this->netMap.size();
  assert(this->nnets > 0); // networks() guarantees it

  // network switch time
  this->netSwitchTime = configFile.uintValue("wichoicemaker", 
                                             "net-switch-time",
                                             NET_SWITCH_TIME_DEF);

  // choice delay
  const unsigned choiceDelayMs = configFile.uintValue("wichoicemaker",
                                                      "choice-delay",
                                                      CHOICE_DELAY_DEF,
                                                      CHOICE_DELAY_MIN,
                                                      CHOICE_DELAY_MAX);
  this->choiceDelay = std::chrono::milliseconds(choiceDelayMs);

  // forecast window size
  this->lookahead = configFile.uintValue("wichoicemaker", 
                                         "lookahead",
                                         LOOKAHEAD_DEF, 
                                         LOOKAHEAD_MIN,
                                         LOOKAHEAD_MAX);

  // initialize transferable data table, which is used by the selection algo
  this->tdata = U64Table(this->lookahead,
                         std::move(U64Vector(this->nnets, 0 /*def*/)));
}

/**
 * Destructor. Needs to be made virtual on abstract classes to ensure proper destruction.
 */
WiChoiceMaker::~WiChoiceMaker() {}

/**
 * Executes a loop that reacts to changes in mobility conditions by computing the interface to use in order
 * to maxime throughput under those new conditions, and writes that choice to the dedicated shared
 * memory region.
 *
 * Overrides Runnable::run().
 * Meant to be run as a thread.
 */
void WiChoiceMaker::run() {
  
  GpsInfoReader gpsInfoReader;   // to read mobility info
  GpsInfo gpsInfo;               // to store read mobility info
  
  // where to write interface choices to
  WiChoiceWriter& wiChoiceWriter = this->getWiChoiceWriter();

  unsigned curNetId = NET_CHOICE_DEF; // 0 is the default network choice
  bool firstIter = true;
  
  // main loop
  while (!this->endProgram.load()) {

    gpsInfoReader.getGpsInfoOnUpdate(gpsInfo); // block until update

    std::this_thread::sleep_for(this->choiceDelay); // give fback time to arrive

    NperfSearchRes nperfSearchRes = this->nperfDb.searchNperf(gpsInfo);
    
    // log decision
    std::stringstream ss;
    ss << "WiChoiceMaker" << this->printTag << ": gpstime=" << gpsInfo.gpstime;
    
    // do we have data to work with?
    if (!nperfSearchRes.first) {
      ss << ", no performance data for current location: skipping.";
      LOG_VERBOSE(ss.str().c_str());
      continue; // no data, can't proceed
    }

    // there is data, so determine best network
    unsigned newNetId = this->findMaxDataNet(curNetId, nperfSearchRes.second);

    // log our decision
    ss << ", curNetId=" << curNetId << ", newNetId=" << newNetId;
    
    // log perf table leading to decision
    for (unsigned netId=0; netId < this->nnets; netId++) {
      ss << std::endl << "PerfSum netId " << netId << ":";

      uint64_t totalData = 0;
      for (unsigned off=0; off < this->lookahead; off++) {
        NperfSums const& nperfSums = nperfSearchRes.second[netId][off];
        const uint64_t cdata = this->getExpTput(nperfSums); // will use mes/est
        ss << " " << cdata;
        
        if ((netId == curNetId && off > 0) || off > this->netSwitchTime)
          totalData += cdata;
        
      }
      ss << " (" << totalData << ")";
    } // net loop end

    // do the actual logging
    LOG_VERBOSE(ss.str().c_str());

    if (newNetId != curNetId || firstIter) { // is there a change?

      assert(newNetId < this->nnets); // can't be out of range
      
      // translate net id to iface and write it to shm
      std::string const& iface = this->netMap[newNetId].iface;
      wiChoiceWriter.setIface(iface);
  
      curNetId = newNetId; // update current
      firstIter = false;   // no longer 1st iteration
    }
  } // main while() end
}

/**
 * Determine and return the network that we should pick at the current time in order to maximize the total
 * amount of data that can be transferred over the entire lookahead window, according to past performance.
 * A dynamic-programming algorithm is used for this purpose.
 *
 * @param curNetId: the currently selected network.
 * @param nperfSumTable: historical performance over the lookahead window for each network.
 * @return the id of the network that maximizes the total amount of transferable data.
 */
unsigned WiChoiceMaker::findMaxDataNet(const unsigned curNetId,
                                       NperfSumTable const& nperfSumTable) {

  // find transferable data for each (time, network) pair
  // note there's no need to reset this->tdata because values are always written
  // before being read

  // for each offset within lookahead (descending order)
  for (int off=this->lookahead-1; off >= 0; off--) {

    // helpers to tell whether there's lookhead window beyond current point
    const unsigned nextOffStay = off + 1;
    const bool hasFutureStay = nextOffStay < this->lookahead;
    const unsigned nextOffSwitch = nextOffStay + this->netSwitchTime;
    const bool hasFutureSwitch = nextOffSwitch < this->lookahead;

    // for each network (order irrelevant)
    for (unsigned netId=0; netId < this->nnets; netId++) {

      // find transferable data for network (default is staying put)
      uint64_t maxFdata = hasFutureStay ? this->tdata[nextOffStay][netId] : 0;
      unsigned nextNetId = netId;

      if (hasFutureSwitch) { // is there window beyond an iface switch?
  
        // try every switching possibility
        for (unsigned onetId=0; onetId < this->nnets; onetId++) {

          if (onetId == netId) continue; // no point in switching to same

          const uint64_t maxFdataSwitch = this->tdata[nextOffSwitch][onetId];
          if (maxFdataSwitch > maxFdata) { // update estimate if new max
            maxFdata = maxFdataSwitch;
            nextNetId = onetId;
          }
        } // end other interface (onetId) loop
      }

      // at this point we have computed maxFdata for network netId
      
      // are we at offset zero or not?
      if (off > 0) { // record transferable data for later use

        // note: nperfSumTable is nets x lookahead (opposite of tdata)
        NperfSums const& nperfSums = nperfSumTable[netId][off];
        const uint64_t cdata = this->getExpTput(nperfSums); // will use mes/est
        this->tdata[off][netId] = cdata + maxFdata;

      } else if (netId == curNetId) return nextNetId; // offset zero stop point

    } // end network (netId) loop
  } // end offset (off) loop
  
  assert(false); // we should've returned before getting here
  
  return curNetId; // never reached
}

// WiChoiceMaker class implementation end

/**
 * Creates and runs all the components that make up the system's WiFi interface select launches the threads 
 * that make up the system's interface selection component:
 *   - Throughput measurements feedback receiver
 *   - Throughput estimator
 *   - Gps logger
 *   - Wi-Fi interface choice maker
 */
int main(int argc, char *argv[]) {
  
  LOG_INIT(LOG_FNAME, "wichoicemaker");

  LOG_VERBOSE("wichoicemaker starting");

  // support for cmd arg to toggle tput estimate printing
  bool printTputEst = false;
  if (argc >= 2) {
    const std::string firstArg = argv[1]; // implicit std::string conversion
    
    if (!(printTputEst = firstArg == "-p" || firstArg == "--print-tput-est"))
      std::cerr << "Invalid argument: " << firstArg
                << ". Only -p/--print-tput-est supported." << std::endl;

    if (argc > 2) // we ignore spurious arguments
      std::cerr << "Ignoring all command-line arguments beyond the first."
                << std::endl;
  }
  
  // create nperf summarization strategy
  std::unique_ptr<NperfSumStrat> nperfSumStratPtr =
                                      std::move(NperfSumStratFactory::create());
  
  // initializes in-memory performance database
  NperfMemDb nperfDb(*nperfSumStratPtr);
  nperfDb.loadFromDisk(); // load previous data if it exists

  // throughput data sources
  FbackReceiver fbreceiver(nperfDb);              // for measurements
  TputEstAggr tputEstAggr(nperfDb, printTputEst); // for estimates

  // wichoice makers
  WiChoiceMakerMes wiChoiceMakerMes(nperfDb, *nperfSumStratPtr); // mes-based
  WiChoiceMakerEst wiChoiceMakerEst(nperfDb, *nperfSumStratPtr); // est-based

  // get runner
  Runner& runner = Runner::getInstance();
  
  // add runnables to runner
  runner.addRunnable(nperfDb);
  runner.addRunnable(fbreceiver);
  runner.addRunnable(tputEstAggr);
  runner.addRunnable(wiChoiceMakerMes);
  runner.addRunnable(wiChoiceMakerEst);

  runner.run(); // actually run them runnables

  nperfDb.saveToDisk(); // save data before end

  LOG_CLOSE();

  return 0;
}
