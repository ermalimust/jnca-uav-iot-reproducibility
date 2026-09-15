/**
 * Class to send throughput feedback from data receiver to data sender.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef FBACK_SENDER_HPP__
#define FBACK_SENDER_HPP__

#include <cstdint> // uint*_t
#include <string>  // std::string
#include <map>     // std::map

#include "../../util/conf/Network.hpp" // type NetworkId
#include "../tput/TputLog.hpp"         // class TputLog
#include "../DataTransfer.hpp"         // class DataTransfer


class FbackSender : public DataTransfer {

public:
  /**
   * Constructor method.
   */
  explicit FbackSender();
  
  /**
   * Records the reception of a certain number of bytes on a given network.
   *
   * @param netId: the id of the network the data was received on.
   * @param nbytes: the number of bytes received.
   * @return true if the data is logged successfully, false if the data is irrelevant for feedback purposes
   *         (i.e., netId is not a valid choice for wichoicemaker).
   */
  bool logRx(const NetworkId netId, const uint32_t nbytes);

protected:
  /**
   * Configures communication according to configuration file.
   */
  void configure() override;
  
  /**
   * Helper method to create and bind the connection socket(s) for communication.
   * Overrides superclass abstract method.
   */
  virtual void initConnSocks() override;
  
  /**
   * Implements a thread that sends, periodically, the (GPS) timestamped number of received bytes on
   * each connection to the feedback receiver.
   */
  void commThread() override;
  
  /**
   * Implemented as a no-op because at this time there's nothing we want to print periodically.
   */
  void printerThread() override;
  
private:
  TputLog tputLog;
  std::map<NetworkId, NetworkId> wcmNetIdMap; // glob net id -> wichoice net id
  uint32_t fbackInterval; // millis
};

#endif // FBACK_SENDER_HPP__
