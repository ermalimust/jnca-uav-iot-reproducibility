/**
 * Implements an abstract class that serves as the basis for concrete network performance database
 * implementations.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <cassert> // assert()
#include <sstream> // std::stringstream

#include "../../../util/log/LogFile.hpp"       // LOG_*
#include "../../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../../dtransfer/tput/Tput.hpp"    // types Tput*
#include "../../../gps/shm/GpsInfoReader.hpp"  // class GpsInfoReader
#include "../../../gps/GpsInfo.hpp"            // struct GpsInfo

#include "NperfDb.hpp" // class NperfDb

/**
 * Constructor method. To be called from concrete subclasses.
 *
 * @param nperfSumStrat: network performance summarization strategy to use.
 */
NperfDb::NperfDb(NperfSumStrat const& nperfSumStrat) :
                                   mobiKeyFactory(),
                                   curMobiKey(this->mobiKeyFactory.createCur()),
                                   nperfSumStrat(nperfSumStrat),
                                   mobiHistMap(),
                                   tputSampQueue() {

  // configure parameters from config file
  WcConfigFile configFile;

  // nnets
  this->nnets = configFile.nnetworks("wichoice");
  assert(this->nnets > 0); // nnetworks() guarantees it

  // forecast window size
  this->lookahead = configFile.uintValue("wichoicemaker", "lookahead",
                                         LOOKAHEAD_DEF, LOOKAHEAD_MIN,
                                         LOOKAHEAD_MAX);
}

// data-retrieval related methods start
/**
 * Returns lookahead window size.
 *
 * @return lookahead window size.
 */
std::size_t NperfDb::getLookahead() const { return this->lookahead; }

/**
 * Retrieves the network performance summary table associated with the mobility parameters passed as
 * an argument. The retrieval may fail due for lack of data associated with the provided mobility.
 *
 * @param mobiKey: mobility information for which we want to retrieve a network performance summary.
 * @return a pair made up of a boolean indicating whether the retrieval was successful or not, and a
 * reference to the requested performance data table, which will only be valid if the boolean is true.
 */
NperfSearchRes NperfDb::searchNperf(GpsInfo const& gpsInfo) {

  // create mobility key out of gps info
  MobiKey mobiKey = this->mobiKeyFactory.create(gpsInfo);

  return this->searchNperf(mobiKey); // delegate to concrete class
}
// data-retrieval related methods end

// data-saving related methods start
/**
 * Adds the provided throughput sample to the database.
 *
 * @param tputSamp: throughput sample to be added to the database.
 * @param tstamp: time at which throughput was collected (secs).
 */
void NperfDb::addTputSamp(TputSamp const& tputSamp, const uint32_t tstamp) {
  this->lockData();

  this->addTputSampLocked(tputSamp, tstamp);
  
  this->unlockData();
}

/**
 * Adds all the throughput samples in the provided map, indexed by timestamp, to the database.
 *
 * @param tputSampMap: map with the samples to be added
 */
void NperfDb::addTputSampMap(TputSampMap const& tputSampMap) {
  
  this->lockData();
  
  // go through each tput sample
  // note: std::map ensures iteration in ascending order
  for (const auto& kvp : tputSampMap) {
    const uint32_t tstamp = kvp.first;
    TputSamp const& tputSamp = kvp.second;
    this->addTputSampLocked(tputSamp, tstamp);
  }

  this->unlockData();
}

/**
 * Helper method that adds a throughput sample to the database
 * It makes the following assumptions:
 *   1. Database has been locked prior to call;
 *   2. Estimates are saved prior to measurements;
 *   3. Method is called with strictly increasing timestamps.
 *
 * @param tputSamp: throughput sample to be added to the database.
 * @param tstamp: time at which throughput was collected (secs).
 */
void NperfDb::addTputSampLocked(TputSamp const& tputSamp,
                                const uint32_t tstamp) {

  // do we have the mobility information needed to process this sample?
  if (this->mobiHistMap.count(tstamp) == 0) { // if not, queue it
    this->tputSampQueue[tstamp].emplace_back(std::move(tputSamp));
    logTputSamp(tputSamp, tstamp, true /*queued*/);
    return;
  }

  logTputSamp(tputSamp, tstamp, false /*queued*/);

  // add tput sample to all mobility keys within lookahead
  // iterate over mobility history (guaranteed ascending order)
  for (auto itr = this->mobiHistMap.begin(), end = this->mobiHistMap.end();
       itr != end; ) {

    const uint32_t tstampHist = itr->first;
    
    if (tstampHist > tstamp) break; // too new, no need to continue
    
    const unsigned offset = tstamp - tstampHist;
    MobiHistEntry& mobiHistEntry = itr->second;

    if (offset < this->lookahead) { // within lookahead
      this->addTputSampLocked(tputSamp, offset, mobiHistEntry.mobiKey);
      ++itr; // advance iterator since we're not erasing
      
    } else { // outside lookahead (too old), may erase
    
      // this source is done with this entry
      mobiHistEntry.doneSources.emplace(tputSamp.source);

      // if all sources done with entry, erase it
      if (mobiHistEntry.doneSources.size() == TputSource::NSOURCES) {

        itr = this->mobiHistMap.erase(itr);
    
        // log erasure
        std::stringstream ss;
        ss << "NperfDb erasing mobiHist tstamp=" << tstampHist 
           << ", map size=" << this->mobiHistMap.size()-1;
        LOG_VERBOSE(ss.str().c_str());
      }
      else ++itr; // otherwise just advance the iterator
    }
  } // mobility history loop end
}

/**
 * Helper method that logs a throughput sample that is being added to the database.
 *
 * @param tputSamp: throughput sample to be logged.
 * @param tstamp: when the data was gathered (secs).
 * @param queued: whether the sample is being added or queued.
*/
inline void NperfDb::logTputSamp(TputSamp const& tputSamp,
                                 const uint32_t tstamp,
                                 const bool queued) {
  
  static const char *tputSourceNames[] = {"estimate","measurement"};
  
  // log what is being saved
  std::stringstream ss;
  ss << "NperfDb::addTputSamp() tstamp=" << tstamp
     << ", source=" << tputSourceNames[tputSamp.source];

  if (queued) ss << ", queued"; // nothing else to say
  else { // general case, log the contents
    ss << ", data: ";
    const std::size_t nnets = tputSamp.tputVec.size();
    if (nnets == 0) ss << "no data"; // just a gap filler
    else {
      for (std::size_t netId=0; netId < nnets; netId++)
        ss << "netId " << netId << "=" << tputSamp.tputVec[netId] << " ";
    }
  }

  LOG_VERBOSE(ss.str().c_str());
}
// data-saving related methods end

/**
 * Constructor method.
 *
 * @param the mobility key for the entry.
 */
NperfDb::MobiHistEntry::MobiHistEntry(MobiKey const& mobiKey) :
                                                    mobiKey(std::move(mobiKey)),
                                                    doneSources() { }

/**
 * Executes a loop that updates mobility history in reaction to changes in gps information.
 *
 * Overrides Runnable::run().
 * Meant to be run as a thread.
 */
void NperfDb::run() {

  // main loop
  GpsInfoReader gpsInfoReader;
  GpsInfo gpsInfo;
  while (!this->endProgram.load()) {

    gpsInfoReader.getGpsInfoOnUpdate(gpsInfo); // block until update

    // process newly-received mobility information
    const uint32_t tstamp = gpsInfo.gpstime;
    MobiKey newMobiKey = this->mobiKeyFactory.create(gpsInfo);

    this->lockData();
   
    if (newMobiKey != this->curMobiKey) {  // did mobiKey change?
      this->resetCurPerfSum(); // clean the slate

      std::stringstream ss;
      ss << "NperfDb mobiKey change: ("
         << this->curMobiKey.toString() << ") -> ("
         << newMobiKey.toString() << ")";
      LOG_VERBOSE(ss.str().c_str());

      this->curMobiKey = newMobiKey; // update current mobiKey
    }

    // add to mobility history
    this->mobiHistMap.emplace(std::piecewise_construct,
                              std::forward_as_tuple(tstamp),
                              std::forward_as_tuple(this->curMobiKey)
                              );

    // is there anything in the tput sample queue that we ned to take care of?
    auto itr = this->tputSampQueue.find(tstamp);
    if (itr != this->tputSampQueue.end()) { // something in queue!
      TputSampVec const& tputSampVec = itr->second; // retrieve the samples
     
      for (auto& tputSamp : tputSampVec) // add all samples in queue
        this->addTputSampLocked(tputSamp, tstamp);

      this->tputSampQueue.erase(itr); // queue entry no longer needed
    }
    this->unlockData();
  }

 // no cleanup needed
}
