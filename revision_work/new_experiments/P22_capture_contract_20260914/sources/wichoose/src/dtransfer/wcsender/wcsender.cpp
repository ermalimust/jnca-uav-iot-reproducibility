/**
 * This program streams data in bulk to a receiver counterpart.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#include <memory>   // std::unique_ptr
#include <utility>  // std::move
#include <iostream> // std::cerr, std::endl

#include "../../util/thread/Runner.hpp" // class Runner
#include "../../util/log/LogFile.hpp"   // LOG_*
#include "DataSender.hpp"               // class DataSender

#define LOG_FNAME "/var/log/wcsender.log"

/**
 * Basically configures and launches two threads: a data receiver, and a feedback sender.
 *  
 * @param argc: number of command line arguments, including the program's name.
 * @param argv: array of command line arguments, as strings.
 * @return 0 on success, other value on error.
 */
int main(int argc, char *argv[]) {

  LOG_INIT(LOG_FNAME, "wcsender");

  LOG_VERBOSE("wcsender starting");
  
  if (argc > 1)
    std::cerr << "Ignoring all command-line arguments." << std::endl;

  // create data sender runnable (move is used to pass responsibility to caller)
  std::unique_ptr<DataSender> dsenderPtr =
                                      std::move(DataSender::createDataSender());

  Runner& runner = Runner::getInstance(); // get runner
  runner.addRunnable(*dsenderPtr);        // add runnable to runner
  runner.run();                           // actually run the runnable

  LOG_CLOSE();

  return 0;
}
