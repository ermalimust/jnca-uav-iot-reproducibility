/**
 * Implement a class that sends data in bulk across the wifi interface deemed to be the best.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <string>     // std::string
#include <sstream>    // std::stringstream
#include <thread>     // std::thread
#include <memory>     // std::unique_ptr, std::make_unique
#include <functional> // std::ref
#include <stdexcept>  // std::invalid_argument

#include "../../wichoice/shm/WiChoiceReaderMes.hpp" // class WiChoiceReaderMes
#include "../../wichoice/shm/WiChoiceReaderEst.hpp" // class WiChoiceReaderEst

#include "DataSenderBest.hpp" // class DataSenderBest

/**
 * Constructor method.
 *
 * @param dataSource: what source to use to pick best interface (measurements or estimates).
 */
DataSenderBest::DataSenderBest(DataSource dataSource) : DataSender() {
  
  // wiChoiceReaderPtr init depends on using measurements or estimates.
  // note: ternary operator won't work, due to different types on each branch
  switch (dataSource) {
    case DataSource::ESTIMATE:
      this->wiChoiceReaderPtr = std::make_unique<WiChoiceReaderEst>();
      break;
    case DataSource::MEASUREMENT:
      this->wiChoiceReaderPtr = std::make_unique<WiChoiceReaderMes>();
      break;
    default:
      throw std::invalid_argument("DataSenderBest(): invalid data source. Only ESTIMATE and MEASUREMENT supported.");
      break;
  }
}

/**
 * Implements a thread that continually sends data at the fastest possible rate over the best interface,
 * according to the wifi choice shared memory.
 */
void DataSenderBest::commThread() {

  this->initConnSocks();

  // initialize best interface - must be done prior to send loop start
  const std::string bestIface = this->wiChoiceReaderPtr->getIfaceNow();

  // best connection pointer - atomic for thread safety
  // std::atomic requires trivially copyable value, hence pointer and not ref
  std::atomic<DataTransfer::ConnInfo*> bestConnPtr;
  bestConnPtr.store(&this->getConnInfo(bestIface));

  // launch thread to monitor changes in interface choice
  // note: std::ref creates wrapper around reference, needed for threads
  std::thread monitorThread(&DataSenderBest::monitorWiChoice, this,
                           std::ref(bestConnPtr));
  
  // loop sending messages over the best connection
  while (!this->endProgram.load()) this->sendMsg(*bestConnPtr.load());
  
  monitorThread.join(); // no need to monitor anything anymore
  
  this->closeConnSocks();
}

/**
 * Monitors changes in interface selection and updates the best connection pointer to reflect them.
 *
 * @param bestConnPtr: the connection pointer to be updated upon interface selection change. It is
 *                    atomic because it needs to be used from multiple threads.
 */
void DataSenderBest::monitorWiChoice(
                            std::atomic<DataTransfer::ConnInfo*>& bestConnPtr){
  
  while(!this->endProgram.load()) {
    
    // block waiting for interface choice change
    const std::string bestIface = this->wiChoiceReaderPtr->getIfaceOnUpdate();
    bestConnPtr.store(&this->getConnInfo(bestIface));
  } // while end
}

/**
 * Halts the runnable. The superclass one is insufficient because the thread may be blocked waiting for
 * the wichoice to be updated. We must therefore trigger that update manually.
 */
void DataSenderBest::stop() {
  this->Runnable::stop();                     // sets endProgram flag to true
  this->wiChoiceReaderPtr->notifyListeners(); // wake listener up
}
