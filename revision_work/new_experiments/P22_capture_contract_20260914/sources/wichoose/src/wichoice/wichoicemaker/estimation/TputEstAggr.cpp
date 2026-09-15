/**
 * Implements class that periodically (1 Hz) estimates network throughput and saves it on a database.
 *
 * Rui Meireles  {@vassar.edu} 2024
 */

#include <cstdint>  // uint*_t
#include <cstddef>  // std::size_t
#include <cassert>  // assert()
#include <iostream> // std::cout

#include "../../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../../util/conf/Network.hpp"      // struct Network, type NetworkId
#include "../../../gps/GpsInfo.hpp"            // struct GpsInfo
#include "../../../gps/shm/GpsInfoReader.hpp"  // class GpsInfoReader
#include "../../../dtransfer/tput/Tput.hpp"    // struct TputSamp
#include "../../chaninfo/ChanInfo.hpp"         // type ChanInfoMap
#include "../../chaninfo/ChanInfoReader.hpp"   // class ChanInfoReader
#include "TputEstimator.hpp"                   // type TputEstimatorPtrVec
#include "TputEstimatorFactory.hpp"            // class TputEstimatorFactory

#include "TputEstAggr.hpp" // class TputEstAggr

/**
 * Constructor.
 *
 * @param nperfDb: the database object to save throughput estimates to.
 * @param printEst: whether the
 */
TputEstAggr::TputEstAggr(NperfDb& nperfDb, const bool printEst) : 
                         nperfDb(nperfDb), netMap(), printEst(printEst) {
  
  // read networks
  WcConfigFile wcConfigFile;
  wcConfigFile.networks("wichoice", this->netMap);
}

/**
 * Executes a loop that periodically estimates network performance and saves it to the associated database.
 *
 * Overrides Runnable::run().
 * Meant to be run as a thread.
 */
void TputEstAggr::run() {
  GpsInfoReader gpsInfoReader; // for reading gps information
  GpsInfo gpsInfo;             // to store the gps info read
  
  ChanInfoReader chanInfoReader; // for reading channel state information
  ChanInfoMap const& chanInfoMap = chanInfoReader.getChanInfoMap();
  
  const std::size_t nnets = this->netMap.size(); // call it only once

  TputEstimatorPtrVec tputEstimatorPtrVec; // for estimating throughput
  TputEstimatorFactory::createVec(this->netMap, tputEstimatorPtrVec);
  assert(tputEstimatorPtrVec.size() == nnets); // 1:1 estimator:network

  // print header if needed
  if (this->printEst) std::cout << "gpstime, iface, nbytesEst" << std::endl;

  // main loop
  while (!this->endProgram.load()) {

    gpsInfoReader.getGpsInfoOnUpdate(gpsInfo); // blocks until gps update
    chanInfoReader.updateChanInfo();           // update channel info map
    
    TputSamp tputSamp(nnets, ESTIMATE); // place to save estimates

    // time to perform estimation;
    for (auto& kvp : this->netMap) { // for each network
      const NetworkId netId = kvp.first;
      Network const& net = kvp.second;
    
      auto chitr = chanInfoMap.find(netId); // retrieve channel info
      const uint32_t tput = chitr == chanInfoMap.end() ? 0 : // default to zero
               tputEstimatorPtrVec[netId]->estimateTput(gpsInfo, chitr->second);

      if (this->printEst) // should the estimate be printed?
        std::cout << gpsInfo.gpstime << ", "
                  << net.iface << ", "
                  << tput << std::endl;
      
      tputSamp.tputVec[netId] = tput; // save estimated throughput
    } // network loop end

    this->nperfDb.addTputSamp(tputSamp, gpsInfo.gpstime); // save tput estimates
  } // main loop end
}
