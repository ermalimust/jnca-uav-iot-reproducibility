/**
 * Implements abstract class that acts as template for classes that send data in bulk over one or more
 * wifi interfaces.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#include <string>        // std::string
#include <sstream>       // std::stringstream
#include <memory>        // std::unique_ptr, std::make_unique
#include <unordered_map> // std::unordered_map

#include "../../util/conf/WcConfigFile.hpp"     // class WcConfigFile
#include "../../util/log/LogFile.hpp"           // LOG_*
#include "DataSenderAll.hpp"                    // class DataSenderAll
#include "DataSenderBest.hpp"                   // class DataSenderBest

#include "DataSender.hpp" // class DataSender

#define SEND_THROUGH_DEF "all"

DataSender::DataSender() : Sender("DataTx") {
  
  // can't be called in superclass cons. as child object init is done after
  this->configure();
}

/**
 * Configures communication according to configuration file.
 */
void DataSender::configure() {

  // leverage superclass for common parameters
  this->DataTransfer::configure("data-transfer"  /* section */,
                                PORT_DATA_TX_DEF /* portTxDef */,
                                PORT_DATA_RX_DEF /* portRxDef */);
}

/**
 * Helper method to create and bind the connection socket(s) for communication.
 * Overrides superclass abstract method.
 */
void DataSender::initConnSocks(){
  // delegate to superclass
  this->Sender::initConnSocks(NodeType::CLIENT); // client sends data
}

/**
 * Helper method that sends a single message (with uninitalized data) over the connection passed as
 * argument.
 *
 * @param cinfo: the connection to send the message on.
 */
void DataSender::sendMsg(DataTransfer::ConnInfo& cinfo) {
  if (sendto(cinfo.sockfd, this->sndBuf, sizeof(this->sndBuf), 0 /*flags*/,
             (const struct sockaddr *) &cinfo.sockaddrDst,
             sizeof(cinfo.sockaddrDst)) == -1)
    
    this->closeConnSocksAndExit("DataSender::sendMsg() sendto()");

  cinfo.nbytesAcc += sizeof(this->sndBuf); // log bytes sent
}

/**
 * Creates correct data sender object, according to configuration file.
 */
std::unique_ptr<DataSender> DataSender::createDataSender() {
  
  std::unique_ptr<DataSender> dataSenderPtr; // return value
  
  // read and set send mode
  WcConfigFile configFile;
  std::string streamMode = configFile.strValue("data-transfer",
                                               "send-through",
                                               SEND_THROUGH_DEF);

  // use a switch to distinguish between the different options
  const static std::unordered_map<std::string,int> streamModeToCase {
                                                                {"all", 1},
                                                                {"best-mes", 2},
                                                                {"best-est", 3}
                                                              };
  switch (streamModeToCase.count(streamMode) ?
          streamModeToCase.at(streamMode) : 0) {
    
    case 0: // not found
      { // needed to constraint stringstream scope
        std::stringstream ss;
        ss << "Config exception: section=data-transfer, value=send-through. "
        << "Invalid value " << streamMode
        << " (only 'all', 'best-mes', and 'best-est' accepted). "
        << "Defaulting to 'all'.";
        LOG_ERR(ss.str().c_str());
      }
      // no break so we fall through to case 1

    case 1: // all
      dataSenderPtr = std::make_unique<DataSenderAll>();
      break;

    case 2: // best-mes
      dataSenderPtr = std::make_unique<DataSenderBest>(
                                       DataSenderBest::DataSource::MEASUREMENT);
      break;
    case 3: // best-est
      dataSenderPtr = std::make_unique<DataSenderBest>(
                                       DataSenderBest::DataSource::ESTIMATE);
  }

  return dataSenderPtr; // return the sender we've just created
}
