/**
 * Defines abstract template class for concrete data/feedback sender classes.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef SENDER_HPP__
#define SENDER_HPP__

#include <string>       // std::string

#include "DataTransfer.hpp" // class DataTransfer

class Sender : public DataTransfer {

protected:
  /**
   * Constructor method. Protected because abstract class.
   * 
   * @param printTag: identifying text to use for printing.
   */
  explicit Sender(std::string const& printTag);

  /**
   * Receives data on potentially multiple interfaces, and deletes its processing to processData().
   * Abstract so must be overridden by concrete subclasses.
   */
  virtual void commThread() = 0;

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
};

#endif // SENDER_HPP__
