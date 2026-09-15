/**
 * Implements abstract template class for concrete data/feedback sender classes.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include "Sender.hpp" // class Senders

/**
 * Constructor method.
 */
Sender::Sender(std::string const& printTag) : DataTransfer(printTag) { }

/**
 * Helper method to create and bind the connection socket(s) for communication.
 * This variant takes in a nodeType (client or server) argument to make it shareable by code running on
 * either node.
 *
 * @param nodeType: whether the code is running on the client or server node.
 */
void Sender::initConnSocks(const NodeType nodeType) {

  // delegate base init to superclass
  this->DataTransfer::initConnSocks(nodeType, CommRole::SENDER);

  // build destination address structures (only used by sender)
  for (auto& kvp : this->connMap) { // for each conn
    
    ConnInfo& cinfo = kvp.second; // retrieve connection info
    
    // the destination ip is the complement of ours
    std::string const& ipDst = nodeType == NodeType::CLIENT ?
                                           cinfo.net.ipSrv : cinfo.net.ipCli;

    this->createSockaddr(ipDst, this->portRx, &cinfo.sockaddrDst);
  } // conn loop end
}
