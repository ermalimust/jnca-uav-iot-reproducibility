/**
 * Class to send throughput feedback from data receiver to data sender.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <cstddef> // std::size_t
#include <cassert> // assert()
#include <sstream> // std::stringstream, std::endl
#include <string>  // std::string
#include <chrono>  // std::chrono
#include <thread>  // std::this_thread
#include <iomanip> // std::setfill, std::setw
#include <memory>  // std::unique_ptr

#include "../../util/log/LogFile.hpp"       // LOG_*
#include "../../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../../util/conf/Network.hpp"      // type NetworkId
#include "../../util/time/TimeUtil.hpp"     // getCurTimeMillis()

#include "FbackSender.hpp" // class FbackSender

#define FBACK_INTERVAL_DEF 1000  // millis
#define FBACK_INTERVAL_MIN 100   // millis
#define FBACK_INTERVAL_MAX 30000 // millis
#define FBACK_WIN_SIZE_DEF 10    // how many timestamps to send, each time

/**
 * Constructor method.
 */
FbackSender::FbackSender() : DataTransfer("FbackTx"),
                             tputLog(),
                             wcmNetIdMap(),
                             fbackInterval(FBACK_INTERVAL_DEF) {

  // can't be called in superclass cons. as child object init is done after
  this->configure();
}

/**
 * Configures communication according to configuration file.
 */
void FbackSender::configure() {
  
  // leverage superclass for common parameters
  this->DataTransfer::configure("tput-fback"      /* section */,
                                PORT_FBACK_TX_DEF /* portTxDef */,
                                PORT_FBACK_RX_DEF /* portRxDef */);

  // read and set feedback interval (millis)
  WcConfigFile configFile;
  this->fbackInterval = configFile.uintValue("tput-fback",
                                             "fback-interval",
                                             FBACK_INTERVAL_DEF,
                                             FBACK_INTERVAL_MIN,
                                             FBACK_INTERVAL_MAX);
  
  // read log-related info
  // log capacity is equal to feedback window size
  const uint8_t fbackWinSize = configFile.ubyteValue("tput-fback",
                                                     "window-size",
                                                     FBACK_WIN_SIZE_DEF);

  
  // create a mapping from global net id to wichoice (tput receiver) net id
  NetworkMap netMap;
  configFile.networks("wichoice", netMap);
  for (auto& kvp : netMap) {
    const NetworkId netIdWcm = kvp.first;
    Network const& net = kvp.second;
    this->wcmNetIdMap.emplace(net.id, netIdWcm); // global -> wcm
  }
  
  // configure tput log with learned info
  this->tputLog.resize(fbackWinSize, netMap.size());
}

/**
 * Records the reception of a certain number of bytes on a given network.
 *
 * @param netId: the id of the network the data was received on.
 * @param nbytes: the number of bytes received.
 * @return true if the data is logged successfully, false if the data is irrelevant for feedback purposes
 *         (i.e., netId is not a valid choice for wichoicemaker).
 */
bool FbackSender::logRx(const NetworkId netId, const uint32_t nbytes) {
  
  // translate network id into id used by wichoicemaker (fback receiver)
  auto itr = this->wcmNetIdMap.find(netId);
  
  bool retval = false;
  if (itr != wcmNetIdMap.end()) { // is there a valid mapping?
    const NetworkId netIdWcm = itr->second;
    retval = this->tputLog.logRx(netIdWcm, nbytes); // actually log reception
  }

  return retval;
}

/**
 * Implements a thread that sends, periodically, the (GPS) timestamped number of received bytes on
 * each connection to the feedback receiver.
 **/
void FbackSender::commThread() {
  
  LOG_MSG("FbackSender commThread() start");

  this->initConnSocks();
  
  // allocate enough memory to write the data rx log to
  const std::size_t blen = (this->tputLog).getSerializedLength();
  std::unique_ptr<uint8_t[]> buffer(new uint8_t[blen]);
  
  while (!this->endProgram.load()) {

    uint64_t startTime = TimeUtil::getSystimeMillis(); // note starting time
    
    std::stringstream ssRxLog; // to hold log data in human-readable format
    
    // serialize rx log - will return true if code is correct
    assert(this->tputLog.serialize(buffer.get(), blen, ssRxLog));
    
    // send out the messages over all feedback interfaces
    for (auto& kvp : this->connMap){
      std::string const& iface = kvp.first;
      DataTransfer::ConnInfo const& connInfo = kvp.second;

      // write verbose info about message being sent
      std::stringstream ss;
      ss << "FbackSender sending msg tstamp=" << startTime << ", iface=" <<
            iface << ":" << std::endl << ssRxLog.str();
      LOG_VERBOSE(ss.str().c_str());

      if (sendto(connInfo.sockfd, buffer.get(), blen, 0,
                  /*TODO:MSG_DONTROUTE*/ /*flags*/
                  (const struct sockaddr*) &(connInfo.sockaddrDst),
                  sizeof(connInfo.sockaddrDst)) == -1){
        this->closeConnSocksAndExit("FbackSender sendto()");
        }
      }

    // how long should we sleep for?
    int64_t sleepTime = TimeUtil::getSystimeMillis() - startTime +
                        fbackInterval;
    if (sleepTime > 0) // don't sleep if running late
      std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
      
    } // while() end
  
    this->closeConnSocks();
}

/**
 * Helper method to create and bind the connection socket(s) for communication.
 * Overrides superclass abstract method.
 */
void FbackSender::initConnSocks(){
  // nothing more to do besides delegating to superclass
  DataTransfer::initConnSocks(NodeType::SERVER, CommRole::SENDER);
}

/**
 * Implemented as a no-op because at this time there's nothing we want to print periodically.
 */
void FbackSender::printerThread() { }
