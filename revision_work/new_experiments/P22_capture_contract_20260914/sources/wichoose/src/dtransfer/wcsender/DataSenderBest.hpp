/**
 * Defines a class to send data in bulk across the wifi interface deemed to be the best.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#ifndef DATA_SENDER_BEST_H__
#define DATA_SENDER_BEST_H__

#include <string> // std::string
#include <atomic> // std::atomic
#include <memory> // std::unique_ptr

#include "../../wichoice/shm/WiChoiceReader.hpp" // class WiChoiceReader
#include "../tput/Tput.hpp"                      // enum TputSource
#include "DataSender.hpp"                        // class DataSender

class DataSenderBest : public DataSender {

public:
  // define a more semantically-meaningful name for TputSource
  typedef TputSource DataSource;

  /**
   * Constructor method.
   *
   * @param dataSource: what source to use to pick best interface (measurements or estimates).
   */
  explicit DataSenderBest(DataSource dataSource);

  /**
   * Halts the runnable. The superclass one is insufficient because the thread may be blocked waiting for
   * the wichoice to be updated. We must therefore trigger that update manually.
   */
  void stop() override;

protected:
  /**
   * Implements a thread that continually sends data at the fastest possible rate over the best interface,
   * according to the wifi choice shared memory.
   */
  void commThread() override;

private:
   std::unique_ptr<WiChoiceReader> wiChoiceReaderPtr;

  /**
   * Monitors changes in interface selection and updates the best connection pointer to reflect them.
   *
   * @param bestConnPtr: the connection pointer to be updated upon interface selection change. It is
   *                    atomic because it needs to be used from multiple threads.
   */
  void monitorWiChoice(std::atomic<DataTransfer::ConnInfo*>& bestConnPtr);
};

#endif // DATA_SENDER_BEST_H__
