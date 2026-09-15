/**
 * Defines abstract template class for concrete data/feedback receiver classes.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef RECEIVER_HPP__
#define RECEIVER_HPP__

#include <cstddef>      // std::size_t
#include <atomic>       // atomic
#include <sys/select.h> // fd_set
#include <string>       // std::string

#include "DataTransfer.hpp" // class DataTransfer

class Receiver : public DataTransfer {

protected:
  std::atomic<int> wakefd; // used to wake potentially-blocked select() call

  /**
   * Constructor method. Protected because abstract class.
   * 
   * @param printTag: identifying text to use for printing.
   */
  explicit Receiver(std::string const& printTag);
  
  /**
   * Helper method to create and bind the connection socket(s) for communication.
   * Abstract so must be overridden by concrete subclasses.
   */
  virtual void initConnSocks() = 0;
  
  /**
   * Helper method to create and bind the connection socket(s) for communication.
   * This variant takes in a nodeType (client or server) argument to make it usable by code running on
   * either node.
   *
   * @param nodeType: whether the code is running on the client or server node.
   */
  void initConnSocks(const NodeType nodeType);
  
  /**
   * Makes the file descript set passed as argument contain the needed socket file descriptors for the purposes
   * for receiving data and being able to terminate the program. And nothing else.
   * Doesn't really need to use a mutex because we'll be reading information that isn't going to be changed
   * anywhere else.
   *
   * @param fdset: pointer to file descriptor set to set up.
   * @return the largest file description in the set (useful for subsequent select() call).
   */
  int setUpSockFdSet(fd_set* fdset);
  
  /**
   * Receives data on potentially multiple interfaces, and deletes its processing to processData().
   */
  virtual void commThread() override;
  
  /**
   * Signals the program to stop executing.
   */
  virtual void stop() override;
  
  /**
   * Process a data message that was received.
   * Must be overridden by concrete subclasses.
   *
   * @param cinfo: connection on which the data was received.
   * @param buffer: pointer to memory containing received data.
   * @param nbytes: number of bytes of data received.
   */
  virtual void processData(DataTransfer::ConnInfo& cinfo,
                           const uint8_t* buffer,
                           const std::size_t nbytes) = 0;
};

#endif // RECEIVER_HPP__
