/**
 * A class that implements a writable timestamped log of the throughput achieved by a set of networks.
 * Networks are identified by their ids.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <arpa/inet.h> // htonl
#include <cstring>     // std::memcpy
#include <cstddef>     // std::size_t
#include <ostream>     // std::ostream, std::endl
 
#include "TputLog.hpp" // class TputLog

/**
 * Creates size zero log.
 */
TputLog::TputLog() : TputLogRo(), gpsInfoReader() { }

/**
 * Resizes log to have the size parameters passed in as arguments.
 *
 * @param capacity: how many timestamps we want to log data for.
 * @param nnets: how many networks do we have to log data for, in each timestamp.
 */
void TputLog::resize(const uint8_t capacity, const uint8_t nnets) {
  this->capacity = capacity;
  this->nnets = nnets;

  this->initZeroLog(); // make data vector reflect new parameters
}

/**
 * Helper that zero-Initializes log according to provided parameters.
 * Assumes instance variables capacity and nnets have been initialized beforehand.
 */
void TputLog::initZeroLog() {

  this->dataVector.resize(this->capacity); // create all needed timepoints
  
  for (auto& tpoint : this->dataVector) // populate each timepoint
    tpoint.tputVec.resize(this->nnets, 0 /*value*/);
}

/**
 * Records the reception of a certain number of bytes on a given network.
 *
 * @param netId: the id of the network the data was received on (must be in 0..nnets-1).
 * @param nbytes: the number of bytes received.
 *
 * @return true if the data is logged successfully, false if not due to an invalid network id.
 */
bool TputLog::logRx(const NetworkId netId, const uint32_t nbytes) {

  // do we have a valid netId?
  if (netId >= this->nnets) return false;
  
  uint32_t gpstime = this->gpsInfoReader.getGpstimeNow();

  this->lockData(); // lock vector access
  
  // has gpstime changed since last time?
  if (gpstime != this->curTstamp ) { // update tstamp, idx, and reset timepoint
    this->curTstamp = gpstime; // update timestamp
    // update write idx using circular arithmetic
    this->curIdx = (this->curIdx + 1) % this->capacity;
    
    // reset timepoint
    TputTpoint& tpoint = this->dataVector[curIdx];
    tpoint.tstamp = this->curTstamp; // timestamp to current
    for (auto& tput : tpoint.tputVec) tput = 0; // counts to zero
  }

  // record received bytes
  this->dataVector[curIdx].tputVec[netId] += nbytes;
    
  this->unlockData(); // unlock vector access

  return true;
}

/**
 * Serializes the log to a buffer, ready to send over the network.
 * Locks data to prevent changes during process.
 *
 * @param buffer: pointer to the memory region to serialize to.
 * @param blen: buffer length.
 *
 * @return true if serialization was successful, false if the buffer was too small.
 */
bool TputLog::serialize(uint8_t* buffer, const std::size_t blen) {
  
  this->lockData();
  bool serSuccess = this->serializeWoLock(buffer, blen);
  this->unlockData();
  
  return serSuccess;
}

/**
 * Serializes the log to a buffer, ready to send over the network.
 * Additionally, if serialization was successful, writes the serialized data to the output stream passed as an
 * argument.
 * Locks data to prevent changes during process.
 *
 * This method can be used instead of calling regular serialize() and writeToStream() to ensure that the
 * data written to the stream is exactly the same as that written to the buffer.
 *
 * @param buffer: pointer to the memory region to serialize to.
 * @param blen: buffer length.
 * @param ostream: output stream to write data to.
 *
 * @return true if serialization was successful, false if the buffer was too small.
 */
bool TputLog::serialize(uint8_t* buffer, const std::size_t blen,
                          std::ostream& ostream) {

  this->lockData();
  bool serSuccess = this->serializeWoLock(buffer, blen);
  // call superclass prevent to prevent relocking
  if (serSuccess) this->TputLogRo::writeToStream(ostream);
  this->unlockData();
  
  return serSuccess;
}

/**
 * Serialize the log to a buffer, ready to send over the network.
 * Helper method that does not lock data.
 *
 * @param buffer: pointer to the memory region to serialize to.
 * @param blen: buffer length.
 *
 * @return true if serialization was successful, false if the buffer was too small.
 */
bool TputLog::serializeWoLock(uint8_t* buffer, const std::size_t blen) {

  // do we have enough space to work with?
  if (blen < this->getSerializedLength()) return false;

  // update and write serialization timestamp
  this->serTstamp = gpsInfoReader.getGpstimeNow();
  uint32_t serTstamp = htonl(this->serTstamp);
  std::memcpy(buffer, &serTstamp, sizeof(serTstamp));
  buffer += sizeof(serTstamp);

  // write window size (force 8-bit because we're not calling hton*)
  std::memcpy(buffer, &this->capacity, sizeof(uint8_t));
  buffer += sizeof(uint8_t);

  // write network count (force 8-bit because we're not calling hton*)
  std::memcpy(buffer, &this->nnets, sizeof(uint8_t));
  buffer += sizeof(uint8_t);

  // iterate over the timepoints and write each one
  for (auto& tpoint : this->dataVector) {

    // write tstamp
    uint32_t ntstamp = htonl(tpoint.tstamp);
    std::memcpy(buffer, &ntstamp, sizeof(uint32_t));
    buffer += sizeof(uint32_t);

    // write all the byte counts
    for (auto tput : tpoint.tputVec) { // in index order, we hope
      uint32_t ntput = htonl(tput);
      std::memcpy(buffer, &ntput, sizeof(uint32_t));
      buffer += sizeof(uint32_t);
    } // tput loop end
  } // timepoint loop end

  return true;
}

/**
 * Write the log's contents to an output stream. Locks data so no changes occur during process.
 *
 * @param ostream: the output stream to write to.
 */
void TputLog::writeToStream(std::ostream& ostream) {
  this->lockData(); // prevent data from changing during write
  this->TputLogRo::writeToStream(ostream); // delegate to superclass
  this->unlockData();
}
