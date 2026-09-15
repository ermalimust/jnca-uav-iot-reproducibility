/**
 * Implements a class that maintains an in-memory database of throughput information indexed by mobility
 * data.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <cstddef>   // std::size_t
#include <fstream>   // std::ofstream, std::ifstream
#include <sstream>   // std::stringstream, std::istringstream, std::endl
#include <string>    // std::getline()
#include <tuple>     // std::forward_as_tuple
#include <utility>   // std::piecewise_construct, std::move
#include <exception> // std::exception

#include "../../../util/log/LogFile.hpp"       // LOG_*
#include "../../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../../dtransfer/tput/Tput.hpp"    // types Tput*
#include "../nperfsum/NperfSum.hpp"            // struct NperfSums,
                                               // type NperfSumTable

#include "NperfMemDb.hpp" // class NperfMemDb

#define DB_FNAME_DEF "/root/wichoice-nperfdb"

/**
 * Constructor method.
 *
 * @param nperfSumStrat: network performance summarization strategy to use.
 */
NperfMemDb::NperfMemDb(NperfSumStrat const& nperfSumStrat) :
                           NperfDb(nperfSumStrat),
                           dataMap(),
                           curNperfSumTable(this->nnets,
                                  std::move(NperfSumVector(this->lookahead))) {

  // configure parameters from config file
  WcConfigFile configFile;

  // db filename
  this->dbFname = configFile.strValue("wichoicemaker", "db-fname",
                                      DB_FNAME_DEF);
}

// data-retrieval related methods start
/**
 * Retrieves the network performance summary table associated with the mobility parameters passed as
 * an argument. The retrieval may fail due for lack of data associated with the provided mobility.
 *
 * @param mobiKey: mobility information for which we want to retrieve a network performance summary.
 * @return a pair made up of a boolean indicating whether the retrieval was successful or not, and a
 * reference to the requested performance data table, which will only be valid if the boolean is true.
 */
NperfSearchRes NperfMemDb::searchNperf(MobiKey const& mobiKey) {

  this->lockData();

  // retrieve historical data
  auto dataMapItr = this->dataMap.find(mobiKey);             // search
  const bool histExists = dataMapItr != this->dataMap.end(); // found some?
  
  // do we have realtime data for this mobiKey?
  const bool isCurMobi = mobiKey == this->curMobiKey;
  
  NperfSumTable resNperfSumTable; // empty by default
  
  if (histExists) { // if there's history, it should be the basis
    resNperfSumTable = dataMapItr->second; // performs copy, history as base
    
    if (isCurMobi) { // replace historical with realtime data whenever available
      
      for (std::size_t netId=0; netId < this->nnets; netId++) {
        for (std::size_t off=0; off < this->lookahead; off++) {
    
          NperfSums& nperfSumsSrc = this->curNperfSumTable[netId][off];
          NperfSums& nperfSumsDst = resNperfSumTable[netId][off];
  
          nperfSumsDst.replaceWithIfNonZero(nperfSumsSrc);
        }
      }
    } // realtime data copy if end
  
  } else if (isCurMobi) { // no history but current mobility
    resNperfSumTable = this->curNperfSumTable; // performs copy
  }

  this->unlockData();

  const bool isSearchSuc = histExists || isCurMobi; // successful search?
  NperfSearchRes searchRes(isSearchSuc, std::move(resNperfSumTable));

  return searchRes;
}
// data-retrieval related methods end


// data-saving related methods start
/**
 * Helper method that adds a throughput sample to a given mobility key.
 * Assumes database has been locked prior to call.
 *
 * @param tputSamp: throughput sample to be added to the database.
 * @param offset: how many secs after mobiKey was observed the data was gathered.
 * @param mobiKey: the mobility key the throughput sample should be associated with.
 *
 * @return true if addition was successful, false if offset larger than lookahead.
 */
bool NperfMemDb::addTputSampLocked(TputSamp const& tputSamp,
                                   const unsigned offset,
                                   MobiKey const& mobiKey) {

  if (offset >= this->lookahead) return false; // would be out of range
    
  // add sample to historical table
  NperfSumTable& histNperfSumTable = this->getHistNperfSumTable(mobiKey);
  this->addTputSampLocked(tputSamp, offset, histNperfSumTable);
  
  // should we add to the current sum table as well?
  if (mobiKey == this->curMobiKey)
    this->addTputSampLocked(tputSamp, offset, curNperfSumTable);

  return true;
}

/**
 * Helper method that adds a throughput sample to a performance summary table.
 * Assumes database has been locked prior to call.
 *
 * @param tputSamp: throughput sample to be added to the database.
 * @param offset: offset to index the table, representing how far into the future (s) data was gathered.
 * @param nperfSumTable: the table onto which the throughput sample is to be added.
 *
 * @return true if addition was successful, false if offset larger than lookahead.
 */
bool NperfMemDb::addTputSampLocked(TputSamp const& tputSamp,
                                   const unsigned offset,
                                   NperfSumTable& nperfSumTable) {

  if (offset >= this->lookahead) return false; // would be out of range

  const std::size_t tputVecLen = tputSamp.tputVec.size();
  
  // go over each interface
  for (std::size_t netId=0; netId < this->nnets; netId++) {
  
    // determine tput while guarding against empty vectors (which mean zero)
    const Tput tput = netId < tputVecLen ? tputSamp.tputVec[netId] : 0;
    
    // get the correct table cell
    NperfSums& nperfSums = nperfSumTable[netId][offset];
  
    // get the correct summary within the cell
    NperfSum& nperfSum = tputSamp.source == TputSource::ESTIMATE ?
                         nperfSums.est : nperfSums.mes;

    this->nperfSumStrat.addTputSamp(tput, nperfSum); // finally add the data

  } // tputVec loop end
  
  return true;
}

/**
 * If the provided mobility key already exists in the historical data map, the table currently associated with the
 * key is returned. If it does not, it is added, in association with a blank performance table. Said table is then
 * returned to the caller.
 *
 * @param mobiKey: the mobility key to index the data map with.
 * @return reference to the performance table associated with mobiKey.
 */
NperfSumTable& NperfMemDb::getHistNperfSumTable(MobiKey const& mobiKey) {

  auto dataMapItr = this->dataMap.find(mobiKey);

  if (dataMapItr == this->dataMap.end()) { // new mobikey
    // add new MobiKey, NperfSumTable key-value pair to the map
    auto emplaceRes = this->dataMap.emplace(std::piecewise_construct,
                                            std::forward_as_tuple(mobiKey),
                                            std::forward_as_tuple(this->nnets,
                                               std::move(NperfSumVector
                                                          (this->lookahead))));
    
    dataMapItr = emplaceRes.first; // emplaceRes is <iterator, bool> pair
  }

  // at this point we're sure the iterator refers to a table, the right one
  NperfSumTable& nperfSumTable = dataMapItr->second; // table to add onto

  return nperfSumTable;
}

/**
 * Resets performance information pertaining to current mobility key.
 * To be called upon switch to new mobility key.
 * Assumes data has been locked prior to call.
 * Overrides abstract method from superclass.
 */
void NperfMemDb::resetCurPerfSum() {

  // reset all the performan
  for (auto& perfSumVector : this->curNperfSumTable)
    for (auto& nperfSums : perfSumVector)
      nperfSums.reset();
  
}
// data-saving related methods end


// (de)serialization methods start
enum ReadState { WPARAM, WMOBIKEY, WIFACE, WOFFSET };

/**
 * Load the database contents from disk.
 */
void NperfMemDb::loadFromDisk() {

  // open up file for reading
  std::ifstream dbfile(this->dbFname, std::ifstream::in);

  // does the file exist and can we read from it?
  if (!dbfile.good()) {
    std::stringstream ss;
    ss << "NperfMemDb loadFromDisk() message: can't open file " <<
          this->dbFname << ". Starting with empty database." << std::endl;
    LOG_MSG(ss.str().c_str());
    return;
  }

  // useful variables for reading loop
  ReadState state = ReadState::WPARAM; // waiting for db parameters
  std::string line;  // individual line buffer
  std::string taway; // throwaway token-consumer string
  NperfSumTable* nperfSumTable=0;
  unsigned netId, offset, nseenIfaces=0, nseenOffsets=0;
  uint32_t nentriesEst, nentriesMes;
  uint64_t tputSumEst, tputSumMes;
  
  this->lockData(); // prevent data from changing during operation
  this->dataMap.clear(); // reset the map
  
  try {
    // read file line by line
    while (std::getline(dbfile, line)) {
      std::istringstream iss(line); // convert to stream
      
      // listen to exceptions
      iss.exceptions(std::istringstream::failbit | std::istringstream::badbit);
      
      switch (state) { // reading state machine
          
        case ReadState::WPARAM: { // waiting for db parameters
        
          // line format: nnets $1 lookahead $2
          std::size_t nnets=0, lookahead=0;
          iss >> taway >> nnets >> taway >> lookahead;
          
          // do db parameters match instance config?
          if (nnets != this->nnets || lookahead != this->lookahead) {
            
            std::stringstream ss;
            ss << "NperfMemDb loadFromDisk() error: parameter mismatch - " <<
            "(nnets, lookahead). Class=(" <<
            this->nnets << ", " << this->lookahead << ")" <<
            ". File=(" << nnets << ", " << lookahead << ")" <<
            std::endl;
            
            this->unlockData();
            LOG_FATAL_EXIT(ss.str().c_str());
          }
          
          state = ReadState::WMOBIKEY; // parameters ok, move on to mobikey
          break;
        }
        case ReadState::WMOBIKEY: { // waiting for mobility key
        
          MobiKey mobiKey(line); // line format is delegated to mobiKey
          nperfSumTable = &this->getHistNperfSumTable(mobiKey);
          
          nseenIfaces = 0; // we are yet to see  any nets for this mobikey
          state = ReadState::WIFACE; // we now need an interface id
          break;
        }
        case ReadState::WIFACE: { // waiting for net id
          
          // line format: net $1
          iss >> taway >> netId;
          nseenIfaces++;
          nseenOffsets = 0;
          state = ReadState::WOFFSET; // we now need offsets to go with
          break;
        }
        case ReadState::WOFFSET: { // waiting for offset entry

          // line format: offset $1 nentriesEst $2 tputSumEst $3 
          //                        nentriesMes $4 tputSumMes $5
          iss >> taway >> offset >> taway >> nentriesEst >> taway >> tputSumEst
                                 >> taway >> nentriesMes >> taway >> tputSumMes;

          // store read data
          NperfSums& nperfSums = (*nperfSumTable)[netId][offset];
          nperfSums.est.nentries = nentriesEst;
          nperfSums.est.tputSum = tputSumEst;
          nperfSums.mes.nentries = nentriesMes;
          nperfSums.mes.tputSum = tputSumMes;

          if (++nseenOffsets == this->lookahead) { // have we seen them all?
            // next state depends on whether they are still nets to see
            state = nseenIfaces == this->nnets ? ReadState::WMOBIKEY : ReadState::WIFACE;
          }
          break;
        }
      } // state machine end
      
    } // while getline end
  } catch (std::exception const& e) { // problem opening or reading from file

    this->unlockData(); // free at last
    std::stringstream ss;
    ss << "NperfMemDb::loadFromDisk() file reading error: " << e.what() <<
          ", line: " << line << std::endl;
    LOG_FATAL_EXIT(ss.str().c_str());
  }

  this->unlockData(); // free at last
  
  dbfile.close();
}

/**
 * Save the entire database onto disk for later use.
 */
void NperfMemDb::saveToDisk() {

  // open up file for writing
  std::ofstream dbfile;
  dbfile.exceptions(std::ofstream::failbit | std::ofstream::badbit);

  this->lockData(); // prevent data from changing during operation

  try {
    dbfile.open(this->dbFname, std::ofstream::out);

    // first line: parameters
    dbfile << "nnets " << this->nnets <<
              " lookahead " << this->lookahead << std::endl;

    // go aver the mobikey -> nperfsumtable map
    for (auto& dataMapKvp : this->dataMap) {
      // write mobikey
      MobiKey const& mobiKey = dataMapKvp.first;
      dbfile << mobiKey.toString() << std::endl;
      
      // write perf sum table
      NperfSumTable const& nperfSumTable = dataMapKvp.second;
      unsigned netId = 0;
      for (NperfSumVector const& nperfSumVector : nperfSumTable) { // each net
        
        // write net
        dbfile << "net " << netId++ << std::endl;
        
        unsigned offset = 0;
        for (NperfSums const& nperfSums : nperfSumVector) { // offset in lookahead
         
          dbfile << "o " << offset++
                 << " ee " << nperfSums.est.nentries
                 << " te " << nperfSums.est.tputSum
                 << " em " << nperfSums.mes.nentries
                 << " tm " << nperfSums.mes.tputSum << std::endl;
        } // end offset loop
      } // end net loop
    } // end dataMap loop

  } catch (std::exception const& e) { // problem opening or writing to file
    
    std::stringstream ss;
    ss << "NperfMemDb saveToDisk() error: " << e.what() << std::endl;
    LOG_ERR(ss.str().c_str());
  }

  this->unlockData(); // free at last
  
  dbfile.close();
}
// (de)serialization methods end
