/**
 * Class that receives data in bulk across potentially many interfaces.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#include <cstddef> // std::size_t

#include "../../util/conf/WcConfigFile.hpp" // class WcConfigFile

#include "DataReceiver.hpp" // class DataReceiver

/**
 * Constructor method.
 *
 * @param fbackSender: throughput feedback sender.
 */
DataReceiver::DataReceiver(FbackSender& fbackSender) : 
                                                      Receiver("DataRx"),
                                                      fbackSender(fbackSender) {
  
  // can't be called in superclass cons. as child object init is done after
  this->configure();
}

/**
 * Configures communication according to configuration file.
 */
void DataReceiver::configure() {
  // delegate to superclass
  this->DataTransfer::configure("data-transfer"  /* section */,
                                PORT_DATA_TX_DEF /* portTxDef */,
                                PORT_DATA_RX_DEF /* portRxDef */);

  // read data-receiver specific configuration parameters from conf file
  WcConfigFile configFile;
}

/**
 * Helper method to create and bind the connection socket(s) for communication.
 * Overrides superclass abstract method.
 */
void DataReceiver::initConnSocks() {
  // delegate to superclass
  this->Receiver::initConnSocks(NodeType::SERVER); // server receives data
}

/**
 * Process a data message that was received.
 *
 * @param cinfo: connection on which the data was received.
 * @param buffer: pointer to memory containing received data.
 * @param nbytes: number of bytes of data received.
 */
void DataReceiver::processData(DataTransfer::ConnInfo& cinfo,
                               const uint8_t* buffer,
                               const std::size_t nbytes) {

  this->fbackSender.logRx(cinfo.net.id, nbytes); // tell fbackSender about rx

  // update byte count for printer thread
  this->lockData();          // prevent data from changing under us
  cinfo.nbytesAcc += nbytes; // record received bytes for printing
  this->unlockData();        // free access up
}
