/**
 * Defines abstract class to act as template for classes that send data in bulk over one or more
 * wifi interfaces.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#ifndef DATA_SENDER_H__
#define DATA_SENDER_H__

#include <string> // std::string
#include <atomic> // std::atomic
#include <memory> // std::unique_ptr

#include "../Sender.hpp"                         // class Sender

#define SND_BUF_LEN 1440 // largest size wo/ fragmentation

class DataSender : public Sender {

public:
  /**
   * Creates correct data sender object, according to configuration file.
   */
  static std::unique_ptr<DataSender> createDataSender();

protected:
  /**
   * Constructor method. Protected because this is an abstract class.
   */
  explicit DataSender(); // explicit prevents implicit typecast

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
   * Helper method that sends a single message (with uninitalized data) over the connection passed as
   * argument.
   *
   * @param cinfo: the connection to send the message on.
   */
  void sendMsg(ConnInfo& cinfo);
  
  /**
   * Performs the actual communication function by sending data over the correct interface(s).
   * Thread suffix to indicate it is meant to be run as a thread.
   * Must be overridden by concrete subclasses.
   */
  virtual void commThread() override = 0;

private:
  char sndBuf[SND_BUF_LEN]; // declared here so there's only one
};

#endif // DATA_SENDER_H__
