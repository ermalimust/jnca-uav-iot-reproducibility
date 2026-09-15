/**
 * Implements a class that sends data in bulk across all available wifi interfaces.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <cstddef> // std::size_t
#include <thread>  // std::thread
#include <memory>  // std::unique_ptr

#include "DataSenderAll.hpp" // class DataSenderAll

DataSenderAll::DataSenderAll() : DataSender() { }

/**
 * Implements threads that continually send data at the fastest possible rate over all wifi interfaces.
 */
void DataSenderAll::commThread() {

  this->initConnSocks();

  const std::size_t nconns = this->connMap.size();
  
  // heap allocation because size only known at runtime
  std::unique_ptr<std::thread[]> sthreads(new std::thread[nconns]);

  unsigned int i=0;
  for (auto& kvp: this->connMap) {
    // & implicitly captures the used variables with automatic storage
    // duration by reference
    sthreads[i++] = std::thread([&]() { // thread implemented as lambda
      while (!this->endProgram.load()) this->sendMsg(kvp.second);
    });

  } // for end

  // now we wait for the streaming threads to end
  for (i=0; i < nconns; i++) sthreads[i].join();

  this->closeConnSocks();
}
