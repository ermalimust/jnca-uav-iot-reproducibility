/**
 * A class that represents a read-only timestamped log of the throughput achieved by a set of networks.
 * Networks are identified by their ids.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef TPUT_LOG_RO_HPP__
#define TPUT_LOG_RO_HPP__

#include <cstddef> // std::size_t
#include <cstdint> // uint*_t
#include <vector>  // std::vector
#include <ostream> // std::ostream

#include "Tput.hpp" // type TputVec

#define TPUT_LOG_TSTAMP_DEF 0 // the timestamp to use when there's no data

// useful type definitions
/**
 * A timestamped data point containing one throughput sample per network.
 */
struct TputTpoint {
  uint32_t tstamp = TPUT_LOG_TSTAMP_DEF; // gps time, in seconds
  TputVec tputVec;                       // tput vector, in bytes
};

typedef std::vector<TputTpoint> TputTpointVec;

class TputLogRo {

public:
  /**
   * Creates throughput log from serialized data passed in as argument.
   *
   * @param buffer: pointer to memory where serialized data resides. It's a non-const pointer to
   * an array of const uint8_t, so the buffer contents can't be changed.
   * @param blen: buffer length, to ensure we don't go over.
   */
  TputLogRo(const uint8_t* buffer, std::size_t blen);
  
  /**
   * Returns the log's internal data, which is stored in a vector of timepoints.
   *
   * @return a reference to the log's internal data vector.
   */
  TputTpointVec const& getDataVector() const;
  
  /**
   * Returns the time at which the log was last serialized. This will also work if the log was built from a
   * previously-serialized version.
   * If no serialization has ever been performed, TPUT\_LOG\_TSTAMP\_DEF is returned.
   *
   * @return the log's last serialization timestamp.
   */
  uint32_t getSerTstamp() const;
  
  /**
   * Calculates and returns the number of bytes the log occupies when serialized.
   *
   * @return the number of bytes the log occupies when serialized.
   */
  std::size_t getSerializedLength() const;
  
  /**
   * Write the log's contents to an output stream. Locks data so no changes occur during process.
   *
   * @param ostream: the output stream to write to.
   */
  virtual void writeToStream(std::ostream& ostream);
  
protected:
  uint8_t capacity;
  uint8_t nnets;

  TputTpointVec dataVector;
  
  uint32_t serTstamp = TPUT_LOG_TSTAMP_DEF; // tstamp of serialization  
  
  /**
   * Empty constructor. Needed by subclass.
   */
  TputLogRo();
  
};

#endif // TPUT_LOG_RO_LOG_HPP__
