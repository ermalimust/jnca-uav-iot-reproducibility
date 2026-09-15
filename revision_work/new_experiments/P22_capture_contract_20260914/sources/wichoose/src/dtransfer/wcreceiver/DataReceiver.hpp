/**
 * Defines class to receive data in bulk across potentially many interfaces.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#ifndef DATA_RECEIVER_HPP__
#define DATA_RECEIVER_HPP__

#include <cstdint> // uint*_t

#include "../Receiver.hpp"  // class Receiver
#include "FbackSender.hpp"  // class FbackSender

class DataReceiver : public Receiver {

public:
  /**
   * Constructor method.
   *
   * @param fbackSender: throughput feedback sender.
   */
  explicit DataReceiver(FbackSender& fbackSender);
  
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

private:
  FbackSender& fbackSender;
};

#endif // DATA_RECEIVER_HPP__
