/**
 * Abstract class DataTransfer provides a template for the creation of programs that send or receive data.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#include <linux/socket.h> // AF_INET, SOCK_DGRAM
#include <arpa/inet.h>    // inet_pton
#include <netinet/in.h>   // struct sockaddr_in
#include <netinet/in.h>   // IPPROTO_UDP
#include <unistd.h>       // close()
#include <cassert>        // assert()
#include <cstring>        // memset()
#include <string>         // std::string
#include <sstream>        // std::stringstream, std::endl
#include <system_error>   // std::system_error
#include <utility>        // std::move

#include "../util/conf/Network.hpp"      // struct Network, type NetworkMap
#include "../util/conf/WcConfigFile.hpp" // class WcConfigFile
#include "../gps/shm/GpsInfoReader.hpp"  // class GpsInfoReader

#include "DataTransfer.hpp" // class DataTransfer

#define UNINITIALIZED_FD -1 // marks file descriptor as uninitialized

/**
 * Constructor method. Protected because this is an abstract class.
 *
 * @param printTag: identifying text to use for printing.
 */
DataTransfer::DataTransfer(std::string const& printTag)
    : printTag(printTag), connMap(), threadVec() { }

/**
 * Constructor method for connection info struct.
 *
 * @param net: the network to forever be associated with the connection.
 */
DataTransfer::ConnInfo::ConnInfo(Network const& net) : net(net) { }

/**
 * Configures communication according to configuration file and provided parameters.
 * It's part of the base class because we were able to find a common core.
 *
 * @param section: data transfer config section name.
 * @param portTxDef: port to use for client if none is found in config.
 * @param portRxDef: port to use for server if none is found in config.
 */
void DataTransfer::configure(std::string const& section,
                             const uint16_t portTxDef,
                             const uint16_t portRxDef) {

  // read configuration
  WcConfigFile configFile;

  // read & set ports
  this->portTx = configFile.port(section, "port-tx", portTxDef);
  this->portRx = configFile.port(section, "port-rx", portRxDef);

  // read network information
  NetworkMap netMap;
  configFile.networks(section, netMap);
  assert(netMap.size() > 0); // guaranteed by networks() implementation

  for (auto& kvp : netMap) { // go through all of them networks
    Network const& net = kvp.second;
    ConnInfo connInfo(net); // create connection info
    connInfo.sockfd = UNINITIALIZED_FD; // prevents accidental closure

    this->connMap.emplace(connInfo.net.iface, std::move(connInfo));
  }
}

/**
 * Helper method to create and bind the connection socket(s) for communication.
 * The socket file descriptors are stored along with the rest of the connection info, in connMap.
 *
 * @param nodeType: whether the code is running on the client or server node.
 * @param commRole: whether we're sending or receiving data in this data transfer.
 */
void DataTransfer::initConnSocks(const NodeType nodeType,
                                 const CommRole commRole) {

  // port depends on role
  const uint16_t port = commRole == CommRole::SENDER ? this->portTx :
                                                       this->portRx;

  // go over all connections and create a socket for each relevant interface
  for (auto& kvp : this->connMap) {
    std::string const& iface = kvp.first;
    ConnInfo& cinfo = kvp.second;
    
    // interface address to use depends on role
    std::string const& ip = nodeType == NodeType::CLIENT ? cinfo.net.ipCli :
                                                           cinfo.net.ipSrv;

    std::stringstream ss1;
    ss1 << "Attaching interface " << iface << " @ " << ip << ":" << port;
    LOG_MSG(ss1.str().c_str());
    
    // create udp socket
    int sockfd;
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
      LOG_FATAL_PERROR_EXIT("socket()");
    
    // build address structure
    struct sockaddr_in sockaddr = { 0 };
    this->createSockaddr(ip, port, &sockaddr);
    
    // bind to local address
    if (bind(sockfd, (struct sockaddr*)&sockaddr, sizeof(sockaddr)) < 0) {
      std::stringstream ss2;
      ss2 << "bind() ip " << ip;
      this->closeConnSocksAndExit(ss2.str());
    }
    
    cinfo.sockfd = sockfd; // save socket file descriptor
    
    // build destination address structure (only used by sender)
    if (commRole == CommRole::SENDER) {
      // the destination ip is the complement of ours
      std::string const& ipDst = nodeType == NodeType::CLIENT ?
                                             cinfo.net.ipSrv : cinfo.net.ipCli;

      this->createSockaddr(ipDst, this->portRx, &cinfo.sockaddrDst);
    }
    
  }  // conn sock creation loop end
}

/**
 * Helper method to create a socket address.
 *
 * @param ip: textual representation of the interface's IP address
 * @param port: port the socket should use
 * @param sockaddr: pointer to the socket address structure we should be filling in
 */
void DataTransfer::createSockaddr(std::string const& ip,
                                  const uint16_t port,
                                  struct sockaddr_in* sockaddr) {
  // prep address structure
  memset(sockaddr, '\0', sizeof(struct sockaddr_in)); // zero out struct
  sockaddr->sin_family = AF_INET;                     // Internet address family
  sockaddr->sin_port = htons(port);                   // port

  // try to convert address string to actual ip address
  int rv = inet_pton(AF_INET, ip.c_str(), &(sockaddr->sin_addr));

  if (rv != 1) { // returns 1 on success, 0 or -1 on failure
    this->closeConnSocks();
    std::stringstream ss;
    ss << "rthread inet_pton() ip " << ip;
    std::string str(ss.str());

    if (rv == 0){ // invalid string
      LOG_FATAL_EXIT(str.c_str());
    } else { // system error, errno will have been set
      assert(rv == -1); // the only other option, according to docs
      LOG_FATAL_PERROR_EXIT(str.c_str());
    }
  }
}

/**
 * Close all open connection sockets.
 */
void DataTransfer::closeConnSocks() {
  // iterate over map and call FD_SET on each socket fd
  for (auto& kvp : this->connMap) {
    int sockfd = (kvp.second).sockfd;
    // only close the ones that have been initialized
    if (sockfd != UNINITIALIZED_FD) close(sockfd);
  }
}

/**
 * Closes any open sockets and stops execution, due to an error described in the argument string.
 *
 * @param errorMsg: a textual description of the error causing the stoppage.
 */
void DataTransfer::closeConnSocksAndExit(std::string const& errorMsg){
  this->closeConnSocks();
  LOG_FATAL_PERROR_EXIT(errorMsg.c_str());
}

/**
 * Returns a pointer to the connection associated with the iface name passed as an argument.
 * Throws an error if the interface is not present in the connection map.
 *
 * @param iface: the name of the interface we want to look up.
 * @return a pointer to the connection associated with the argument interface.
 */
DataTransfer::ConnInfo& DataTransfer::getConnInfo(std::string const& iface) {

  // make sure iface exists
  if (this->connMap.count(iface) == 0) {
    std::stringstream ss;
    ss << "Unknown interface " << iface << ".";
    this->closeConnSocksAndExit(ss.str().c_str());
  }

  return this->connMap.at(iface);
}

/**
 * Launches the threads that do actual work.
 */
void DataTransfer::run() {
  
  try {
    // create worker threads
    this->threadVec.emplace_back(&DataTransfer::printerThread, this);
    this->threadVec.emplace_back(&DataTransfer::commThread, this);
    
    // wait for all them threads to finish
    for (auto& thread: this->threadVec) thread.join();

  } catch (const std::system_error& e) {
    std::stringstream ss;
    ss << "DataTransfer::run() thread exception: " << e.what();
    this->closeConnSocksAndExit(ss.str().c_str());
  }
}

/**
 * Prints amount of data sent or received on each connection during each individual second.
 * Printing is triggered by a GPS info change notification.
 */
void DataTransfer::printerThread() {

  GpsInfoReader gpsInfoReader; // to read gps time

  // print header
  std::cout << "gpstime, iface, nbytes" << this->printTag << std::endl;
  uint32_t gpstimeNow = 0, gpstimePrev = 0;
  while (!this->endProgram.load()) {  // for ever, and ever, and ever

    gpstimePrev = gpstimeNow; // yesterday's news
    gpstimeNow = gpsInfoReader.getGpstimeOnUpdate(); // wait until time changes
        
    // iterate over connection map and print stats for each
    for (auto& kvp : this->connMap) {
      std::string const& iface = kvp.first;
      ConnInfo& cinfo = kvp.second;
    
      if (gpstimePrev != 0) // skip 1st iter to eliminate partial seconds
        std::cout << gpstimePrev << ", "
                  << iface << ", "
                  << cinfo.nbytesAcc << std::endl;

      cinfo.nbytesAcc = 0;  // reset byte counter for next round
    }
  } // while end

  // nothing to clean up, just exit
}
