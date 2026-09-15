/**
 * Implements class that receive throughput feedback data across potentially many interfaces and save it onto
 * a network performance database.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#include <cstddef> // std::size_t
#include <cstdint> // uint*_t
#include <sstream> // std::stringstream, std::endl
#include <tuple>   // std::forward_as_tuple
#include <utility> // std::piecewise_construct, std::move

#include "../../../util/log/LogFile.hpp"         // LOG_*
#include "../../../dtransfer/tput/TputLogRo.hpp" // class TputLog
#include "../../../dtransfer/tput/Tput.hpp"      // type TputSource
#include "../../../util/conf/WcConfigFile.hpp"   // class WcConfigFile

#include "FbackReceiver.hpp" // class FbackReceiver

#define TSTAMP_ACCEPT_MIN_DEF 0

/**
 * Constructor method.
 *
 * @param nperfDb: database to save throughput measurements to.
 */
FbackReceiver::FbackReceiver(NperfDb& nperfDb) : 
                                        Receiver("FbackRx"),
                                        nperfDb(nperfDb),
                                        tstampAcceptMin(TSTAMP_ACCEPT_MIN_DEF) {
  
  // can't be called in superclass cons. as child object init is done after
  this->configure();
}

/**
 * Configures communication according to configuration file.
 */
void FbackReceiver::configure() {
  // delegate to superclass
  this->DataTransfer::configure("tput-fback"      /* section */,
                                PORT_FBACK_TX_DEF /* portTxDef */,
                                PORT_FBACK_RX_DEF /* portRxDef */);
}

/**
 * Helper method to create and bind the connection socket(s) for communication.
 * Overrides superclass abstract method.
 */
void FbackReceiver::initConnSocks() {
  // delegate to superclass
  this->Receiver::initConnSocks(NodeType::CLIENT); // client receives fbackcd 
}

/**
 * Process a feedback message that was received.
 *
 * @param cinfo: connection on which the data was received.
 * @param buffer: pointer to memory containing received data.
 * @param nbytes: number of bytes of data received.
 */
void FbackReceiver::processData(DataTransfer::ConnInfo& cinfo,
                                const uint8_t* buffer,
                                std::size_t nbytes) {

  TputLogRo tputLog(buffer, nbytes); // deserialize data into rx log

  // log the reception
  std::stringstream ss;
  ss << "FbackReceiver received message:" << std::endl;
  tputLog.writeToStream(ss);
  LOG_VERBOSE(ss.str().c_str());
  
  TputSampMap tputSampMap; // to store tput measurements
  
  // extract log contents
  TputTpointVec const& tputTpointVec = tputLog.getDataVector();
  const uint32_t serTstamp = tputLog.getSerTstamp(); // when data was written
  
  for (auto& tputTpoint : tputTpointVec) { // iterate over rx data time points
    const uint32_t tstamp = tputTpoint.tstamp;

    // guard against:
    //   1. lack of data (tstamp = 0)
    //   2. duplicate data (tstamp < tstampAcceptMin)
    //   3. data from potentially incomplete tstamps (tstamp == serTstamp)
    if (tstamp != 0 && tstamp >= this->tstampAcceptMin && tstamp != serTstamp) {
  
      // add entry to tput sample map
      tputSampMap.emplace(std::piecewise_construct,
                          std::forward_as_tuple(tstamp),
                          std::forward_as_tuple(
                               std::move(tputTpoint.tputVec),
                               TputSource::MEASUREMENT)
                          );
    }

  } // datarx timepoint loop end

  if (tputSampMap.size() > 0) { // is there new data?
    this->fillTimeGap(tputSampMap);        // ensure no time gaps
    this->nperfDb.addTputSampMap(tputSampMap); // save data to database
  
    // update min tstamp for the future
    // std::maps are sorted, so the largest key is at the end
    const uint32_t tstampMax = tputSampMap.crbegin()->first;
    this->tstampAcceptMin = tstampMax + 1; // nothing earlier matters now
  }
}

/**
 * Helper method that fills in any time gap between previous feedback and the one provided as an
 * argument with empty throughput vector entries.
 *
 * @param tputSampMap: the throughput sample map to be filled.
 */
void FbackReceiver::fillTimeGap(TputSampMap& tputSampMap) {

  // We assume that the received feedback represents a contiguous time
  // slice, which should be the case from the way the system is designed.
  // Thus, if there is a gap, it must be between the previous piece of
  // feedback and the current one.
  
  // no gap filling on the first piece of feedback we get
  if (this->tstampAcceptMin == TSTAMP_ACCEPT_MIN_DEF) return;

  // fill in any time gaps between this and prior feedback
  // std::maps are sorted, so the smallest key is at the start
  const uint32_t tstampMin = tputSampMap.crbegin()->first;
  // add any missing keys
  for (uint32_t tstamp = this->tstampAcceptMin; tstamp < tstampMin; tstamp++) {
    tputSampMap.emplace(std::piecewise_construct,
                        std::forward_as_tuple(tstamp),
                        std::forward_as_tuple(0, TputSource::MEASUREMENT)
                        );
  }
}

/**
 * Implemented as a no-op because at this time there's nothing we want to print periodically.
 */
void FbackReceiver::printerThread() { }
