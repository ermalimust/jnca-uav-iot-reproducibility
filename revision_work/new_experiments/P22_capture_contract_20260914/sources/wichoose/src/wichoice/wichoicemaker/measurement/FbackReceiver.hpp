/**
 * Defines class to receive throughput feedback data across potentially many interfaces and save it onto
 * a network performance database.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#ifndef FBACK_RECEIVER_HPP__
#define FBACK_RECEIVER_HPP__

#include <cstddef> // std::size_t
#include <cstdint> // std::uint*_t

#include "../../../util/conf/Network.hpp"  // type NetworkMap
#include "../../../dtransfer/Receiver.hpp" // class Receiver
#include "../database/NperfDb.hpp"         // class NperfDb

class FbackReceiver : public Receiver {

public:
  /**
   * Constructor method.
   *
   * @param nperfDb: database to save throughput measurements to.
   */
    FbackReceiver(NperfDb& nperfDb);

protected:
  /**
   * Configures communication according to configuration file.
   */
  void configure() override;
  
  /**
   * Helper method to create and bind the connection socket(s) for communication.
   * Overrides superclass abstract method.
   */
  void initConnSocks() override;
  
  /**
   * Process a data message that was received.
   *
   * @param cinfo: connection on which the data was received.
   * @param buffer: pointer to memory containing received data.
   * @param nbytes: number of bytes of data received.
   */
  void processData(DataTransfer::ConnInfo& cinfo,
                   const uint8_t* buffer,
                   const std::size_t nbytes) override;

  /**
   * Implemented as a no-op because at this time there's nothing we want to print periodically.
   */
  void printerThread() override;
  
private:
  NperfDb& nperfDb;         // database to save incoming tput measurements to
  NetworkMap netMap;        // relevant networks: net id (local) -> network
  uint32_t tstampAcceptMin; // smallest timestamp we'll accept data about

  /**
   * Helper method that fills in any time gap between previous feedback and the one provided as an
   * argument with empty throughput entries.
   *
   * @param tputSampMap: the throughput sample map to be filled.
   */
  void fillTimeGap(TputSampMap& tputSampMap);
};

#endif // FBACK_RECEIVER_HPP__
