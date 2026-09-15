/**
 * A class that implements  a read-only timestamped log of the throughput achieved by a set of networks.
 * Networks are identified by their ids.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <arpa/inet.h> // ntohl
#include <cstring>     // std::memcpy
#include <cstddef>     // std::size_t
#include <sstream>     // std::stringstream, std::endl

#include "../../util/log/LogFile.hpp" // LOG_*
 
#include "TputLogRo.hpp" // class TputLogRo

/**
 * Empty constructor. Needed by subclass.
 */
TputLogRo::TputLogRo() : capacity(0), nnets(0), dataVector() { }

/**
 * Creates throughput  log from serialized data passed in as argument.
 *
 * @param buffer: pointer to memory where serialized data resides. It's a non-const pointer to
 * an array of const uint8_t, so the buffer contents can't be changed.
 * @param blen: buffer length, to ensure we don't go over.
 */
TputLogRo::TputLogRo(const uint8_t* buffer, const std::size_t blen) {

  // read serialization timestamp
  std::memcpy(&this->serTstamp, buffer, sizeof(this->serTstamp));
  buffer += sizeof(this->serTstamp);
  this->serTstamp = ntohl(this->serTstamp); // net -> host byte-order conversion

  // read window size (force 8-bit because we're not calling ntoh*)
  std::memcpy(&this->capacity, buffer, sizeof(uint8_t));
  buffer += sizeof(uint8_t);

  // read network count (force 8-bit because we're not calling ntoh*)
  std::memcpy(&this->nnets, buffer, sizeof(uint8_t));
  buffer += sizeof(uint8_t);
  
  // is the buffer long enough for the specified capacity and #networks?
  const std::size_t serializedLen = this->getSerializedLength();
  if (blen < serializedLen){
    std::stringstream ss;
    ss << "Error deserializing tput feedback msg: length " << blen
       << " < " << serializedLen
       << " bytes required for message with capacity "
       << (unsigned) this->capacity << " and "
       << (unsigned) this->nnets << " networks.";
    LOG_FATAL_EXIT(ss.str().c_str());
  }

  // create correct number of timepoints
  this->dataVector.resize(this->capacity);

  // iterate through all the timepoints
  for (std::size_t i=0; i < this->capacity; i++) {
    TputTpoint& tpoint = this->dataVector[i];

    // read timestamp
    std::memcpy(&tpoint.tstamp, buffer, sizeof(uint32_t));
    buffer += sizeof(uint32_t);
    tpoint.tstamp = ntohl(tpoint.tstamp); // net -> host byte-order conversion

    // iterate through all the throughputs
    uint32_t tput;
    for (std::size_t j=0; j < this->nnets; j++) {

      std::memcpy(&tput, buffer, sizeof(uint32_t));
      buffer += sizeof(uint32_t);
      tput = ntohl(tput); // net -> host byte-order conversion
  
      tpoint.tputVec.push_back(tput); // save individual tput
    } // throughput loop end
  } // timepoint loop end
}

/**
 * Returns the log's internal data, which is stored in a vector of timepoints.
 *
 * @return a reference to the log's internal data vector.
 */
TputTpointVec const& TputLogRo::getDataVector() const {
  return this->dataVector;
}

/**
 * Returns the time at which the log was last serialized. This will also work if the log was built from a
 * previously-serialized version.
 * If no serialization has ever been performed, TPUT\_LOG\_TSTAMP\_DEF is returned.
 *
 * @return the log's last serialization timestamp.
 */
uint32_t TputLogRo::getSerTstamp() const { return this->serTstamp; }

/**
 * Calculates and returns the number of bytes the log occupies when serialized.
 *
 * @return the number of bytes the log occupies when serialized.
 */
std::size_t TputLogRo::getSerializedLength() const {
  return sizeof(this->serTstamp) +
         sizeof(uint8_t) /* capacity */ +
         sizeof(uint8_t) /* nnets */ +
         this->capacity * (sizeof(uint32_t) /*tstamp*/ +
                           this->nnets * sizeof(uint32_t) /*tputs*/);
}

/**
 * Write the log's contents to an output stream, without locking the data.
 * Helper method that does not lock data.
 *
 * @param ostream: the output stream to write to.
 */
void TputLogRo::writeToStream(std::ostream& ostream) {
  
  ostream << "TputLog serTstamp=" << this->serTstamp << std::endl;
  
  // iterate over the timepoints and write each one
  for (auto& tpoint : this->dataVector) {

    ostream << "TputLog tstamp=" << tpoint.tstamp << ": "; // write tstamp
    
    // write all the byte counts
    int id = 0;
    for (auto& tput : tpoint.tputVec) // in index-order, we hope
      ostream << "id " << id++ << "=" << tput << " ";
    
    ostream << std::endl; // new line for each timestamp
  }
}
