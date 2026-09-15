/**
 * Abstract class DataTransfer provides a template for the creation of programs that send or receive data.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#ifndef DATA_TRANSFER_HPP__
#define DATA_TRANSFER_HPP__

#include <cstdint>      // uint*_t
#include <netinet/in.h> // struct sockaddr_in
#include <string>       // std::string
#include <thread>       // std::thread
#include <map>          // std::map
#include <vector>       // std::vector

#include "../util/thread/Runnable.hpp" // class Runnable
#include "../util/thread/Lockable.hpp" // class Lockable
#include "../util/conf/Network.hpp"    // type Network

// note: slightly awkward to have these here, but it's the only header shared
// by both data- and feedback-related classes
#define PORT_DATA_TX_DEF 44443
#define PORT_DATA_RX_DEF 44444
#define PORT_FBACK_TX_DEF 44445
#define PORT_FBACK_RX_DEF 44446

class DataTransfer : public Runnable, public Lockable {

public:
  /**
   * Launches the threads that do actual work.
   */
  virtual void run() override;

protected:
  // aux structs and types
  struct ConnInfo { // connection information
    const Network net;
    int sockfd;
    struct sockaddr_in sockaddrDst; // only used by sender
    uint64_t nbytesAcc = 0;
    
    /**
     * Constructor method for connection info struct.
     *
     * @param net: the network to forever be associated with the connection.
     */
    ConnInfo(Network const& net);
  };

  typedef std::map<std::string, ConnInfo> ConnInfoMap; // ifaceName -> connInfo

  enum NodeType {CLIENT, SERVER};   // distinguish client and server nodes
  enum CommRole {SENDER, RECEIVER}; // distinguish sender and receiver roles

  const std::string printTag; // identifying text to use for printing

  ConnInfoMap connMap; // net id -> conn info

  // ports are the same for all connections, so we store them in a single place
  uint16_t portTx{}; // client (sender) port
  uint16_t portRx{}; // server (receiver) port

  std::vector<std::thread> threadVec; // the threads to be executed

  /**
   * Constructor method. Protected because this is an abstract class.
   * Note: explicit keyword prevents automatic type conversion for types featuring constructors callable with
   * a single argument. E.g., prevents dataTransfer == "someString" from compiling,
   *
   * @param printTag: identifying text to use for printing.
   */
  explicit DataTransfer(std::string const& printTag);

  /**
   * Configures communication according to configuration file and provided parameters.
   * It's part of the base class because we were able to find a common core.
   *
   * @param section: data transfer config section name.
   * @param portTxDef: port to use for client if none is found in config.
   * @param portRxDef: port to use for server if none is found in config.
   */
  void configure(std::string const& section,
                 const uint16_t portTxDef,
                 const uint16_t portRxDef);

  /**
   * Configures communication according to configuration file.
   * Subclasses must override, calling the version above with the right arguments.
   */
  virtual void configure() = 0;

  /**
   * Performs the actual communication function by either sending or receiving data.
   * Thread suffix to indicate it is meant to be run as a thread.
   * Must be overridden by concrete subclasses.
   */
  virtual void commThread() = 0;

  /**
   * Meant to print information pertaining to the program's operation.
   * Thread suffix to indicate it is meant to be run as a thread.
   *
   * The default implementation prints, once a second, the amount of data sent/received on each connection.
   * Can be overridden by subclasses, e.g., to suppress printing.
   */
  virtual void printerThread();

  /**
   * Helper method to create and bind the connection socket(s) for communication.
   * Subclasses should override, leveraging the version that has a role argument in the implementation.
   */
  virtual void initConnSocks() = 0;

  /**
   * Helper method to create and bind the connection socket(s) for communication.
   * The socket file descriptors are stored along with the rest of the connection info, in connMap.
   *
   * @param nodeType: whether the code is running on the client or server node.
   * @param commRole: whether we're sending or receiving data in this data transfer.
   */
  void initConnSocks(const NodeType nodeType, const CommRole commRole);

  /**
   * Helper method to create a socket address.
   *
   * @param ip: textual representation of the interface's IP address
   * @param port: port the socket should use
   * @param sockaddr: pointer to the socket address structure we should be filling in
   */
  void createSockaddr(std::string const& ip,
                      const uint16_t port,
                      struct sockaddr_in* sockaddr);

  /**
   * Close all open connection sockets.
   */
  void closeConnSocks();
  
  /**
   * Closes any open sockets and stops execution, due to an error described in the argument string.
   *
   * @param errorMsg: a textual description of the error causing the stoppage.
   */
  void closeConnSocksAndExit(std::string const& errorMsg);
  
  /**
   * Returns a pointer to the connection associated with the iface name passed as an argument.
   * Throws an error if the interface is not present in the connection map.
   *
   * @param iface: the name of the interface we want to look up.
   * @return a pointer to the connection associated with the argument interface.
   */
  DataTransfer::ConnInfo& getConnInfo(std::string const& iface);
};

#endif // DATA_TRANSFER_HPP__
