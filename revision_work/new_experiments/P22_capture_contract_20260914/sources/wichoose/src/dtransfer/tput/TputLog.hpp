/**
 * A class that represents a writable timestamped log of the throughput achieved by a set of networks.
 * Networks are identified by their ids.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef TPUT_LOG_HPP__
#define TPUT_LOG_HPP__

#include <cstddef> // std::size_t
#include <cstdint> // uint*_t
#include <ostream> // std::ostream

#include "../../util/conf/Network.hpp"     // type NetworkId
#include "../../gps/shm/GpsInfoReader.hpp" // class GpsInfoReader
#include "../../util/thread/Lockable.hpp"  // class Lockable
#include "TputLogRo.hpp"                   // class TputLogRo

class TputLog : public TputLogRo, public Lockable {

public:
  /**
   * Creates size zero log.
   */
  TputLog();

  /**
   * Resizes log to have the size parameters passed in as arguments.
   *
   * @param capacity: how many timestamps we want to log data for.
   * @param nnets: how many networks do we have to log data for, in each timestamp.
   */
  void resize(const uint8_t capacity, const uint8_t nnets);
  
  /**
   * Records the reception of a certain number of bytes on a given network.
   *
   * @param netId: the id of the network the data was received on (must be in 0..nnets-1).
   * @param nbytes: the number of bytes received.
   *
   * @return true if the data is logged successfully, false if not due to an invalid network id.
   */
  bool logRx(const NetworkId netId, const uint32_t nbytes);
  
  /**
   * Serializes the log to a buffer, ready to send over the network.
   * Locks data to prevent changes during process.
   *
   * @param buffer: pointer to the memory region to serialize to.
   * @param blen: buffer length.
   *
   * @return true if serialization was successful, false if the buffer was too small.
   */
  bool serialize(uint8_t* buffer, const std::size_t blen);
  
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
  bool serialize(uint8_t* buffer, const std::size_t blen,
                 std::ostream& ostream);
  
  /**
   * Write the log's contents to an output stream. Locks data so no changes occur during process.
   *
   * @param ostream: the output stream to write to.
   */
  void writeToStream(std::ostream& ostream) override;
  
private:
  uint8_t curIdx = 0;
  uint32_t curTstamp = TPUT_LOG_TSTAMP_DEF;

  GpsInfoReader gpsInfoReader; // only needed by writable logs

  /**
   * Helper that zero-Initializes log according to provided parameters.
   * Assumes instance variables capacity and nnets have been initialized beforehand.
   */
  void initZeroLog();

  /**
   * Serializes the log to a buffer, ready to send over the network.
   * Helper method that does not lock data.
   *
   * @param buffer: pointer to the memory region to serialize to.
   * @param blen: buffer length.
   *
   * @return true if serialization was successful, false if the buffer was too small.
   */
  bool serializeWoLock(uint8_t* buffer, const std::size_t blen);
};

#endif // TPUT_LOG_HPP__
