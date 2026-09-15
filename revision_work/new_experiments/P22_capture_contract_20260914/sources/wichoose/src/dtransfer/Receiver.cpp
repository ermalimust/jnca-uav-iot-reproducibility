/**
 * Implements abstract template class for concrete data/feedback receiver classes.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <sys/socket.h>  // recv()
#include <sys/select.h>  // select()
#include <sys/eventfd.h> // eventfd(), eventfd_write()
#include <fcntl.h>       // fcntl(), F_SETFL, etc
#include <cstddef>       // std::size_t

#include "../util/log/LogFile.hpp" // LOG_*

#include "Receiver.hpp" // class Receiver

#define RCV_BUF_LEN 16384

/**
 * Constructor method.
 */
Receiver::Receiver(std::string const& printTag) : DataTransfer(printTag) {
                     
  // create fd to wake up thread blocked on select when it's time to end
  int wakefd = eventfd(0, 0);
  if (wakefd == -1) LOG_FATAL_PERROR_EXIT("Receiver::run() eventfd()");
  this->wakefd.store(wakefd); // store the file descriptor
}

/**
 * Helper method to create and bind the connection socket(s) for communication.
 * This variant takes in a nodeType (client or server) argument to make it shareable by code running on
 * either node.
 *
 * @param nodeType: whether the code is running on the client or server node.
 */
void Receiver::initConnSocks(const NodeType nodeType) {

  // delegate base init to superclass
  this->DataTransfer::initConnSocks(nodeType, CommRole::RECEIVER);
  
  // make sockets non binding, which receivers need
  for (auto& kvp : this->connMap) {
    int sockfd = kvp.second.sockfd;
    if (fcntl(sockfd, F_SETFL, fcntl(sockfd, F_GETFL, 0) | O_NONBLOCK) < 0)
      this->closeConnSocksAndExit("fcntl() setting non-blocking");
  }
}

/**
 * Makes the file descript set passed as argument contain the needed socket file descriptors for the purposes
 * for receiving data and being able to terminate the program. And nothing else.
 * Doesn't really need to use a mutex because we'll be reading information that isn't going to be changed
 * anywhere else.
 *
 * @param fdset: pointer to file descriptor set to set up.
 * @return the largest file description in the set (useful for subsequent select() call).
 */
int Receiver::setUpSockFdSet(fd_set* fdset) {
  FD_ZERO(fdset); // start with a clean slate
  
  int maxfd = this->wakefd.load();
  FD_SET(maxfd, fdset); // needed for end-program wakeup
  
  // iterate over map and call FD_SET on each socket fd
  for (auto& kvp : this->connMap) {
    const int sockfd = kvp.second.sockfd;
    if (sockfd > maxfd) maxfd = sockfd; // keep track of maximum
    FD_SET(sockfd, fdset);
  }

  return maxfd;
}

/**
 * Receives data on potentially multiple interfaces, and deletes its processing to processData().
 */
void Receiver::commThread() {
  
  LOG_MSG("Receiver commThread() start");
  
  // go over all connections and create a socket for each of them
  this->initConnSocks();

  // main server loop
  uint8_t rcvBuf[RCV_BUF_LEN];
  std::size_t nbytes; // counts number of received bytes
  fd_set sockfdSet; // file descriptor set for select
  struct timeval timeout{};
  
  while (!this->endProgram.load()) { // for ever, and ever, and ever

    // set all file descriptors of interest
    int maxfd = this->setUpSockFdSet(&sockfdSet);

    // must reinitialize timeout prior to each select call
    timeout.tv_sec  = 10; // 10 seconds
    timeout.tv_usec = 0;
    
    // wait for there to be something for us to read
    // returns # of ready fds, but we don't need it
    // note need for maxfd + 1
    if (select(maxfd+1, &sockfdSet /*readfds*/, NULL /*writefds*/,
               NULL /*exceptfds*/, &timeout) == -1)
      this->closeConnSocksAndExit("Receiver select()");

    // iterate over connections and check whether we've received anything
    for (auto& kvp : this->connMap) {
      
      DataTransfer::ConnInfo &cinfo = kvp.second;
      const int sockfd = cinfo.sockfd;
      
      if (FD_ISSET(sockfd, &sockfdSet)) {
        if ((nbytes = recv(sockfd, rcvBuf, RCV_BUF_LEN, 0)) > 0) {
          // NOTE: per recv man page, since sockfd is a datagram socket
          // (SOCK_DGRAM), a single and complete message is read at once
          this->processData(cinfo, rcvBuf, nbytes); // process received data
        }
      }
    } // connection loop end

  } // while !endProgram loop end

  this->closeConnSocks(); // clean up and be done
}

/**
 * Signals the program to stop executing.
 */
void Receiver::stop() {
  this->Runnable::stop();                // sets endProgam to true
  eventfd_write(this->wakefd.load(), 1); // wakes up potentially-blocked select
}
