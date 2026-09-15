/**
 * This program serves two purposes:
 *   1. Receive bulk data from a sender counterpart.
 *   2. Send feedback to the data sender,, regarding how much data was successfully received.
 *
 * Rui Meireles (@vassar.edu) 2021, 2024
 */

#include <iostream> // std::cerr, std::endl

#include "../../util/thread/Runner.hpp" // class Runner
#include "../../util/log/LogFile.hpp"   // LOG_*

#include "DataReceiver.hpp" // class DataReceiver
#include "FbackSender.hpp"  // class FbackSender

#define LOG_FNAME "/var/log/wcreceiver.log"

/**
 * Basically configures and launches two threads: a data receiver, and a feedback sender.
 *
 * @param argc: number of command line arguments, including the program's name.
 * @param argv: array of command line arguments, as strings.
 * @return 0 on success, other value on error.
 */
int main(int argc, char *argv[]) {

  LOG_INIT(LOG_FNAME, "wcreceiver");

  LOG_VERBOSE("wcreceiver starting");

  if (argc > 1)
    std::cerr << "Ignoring all command-line arguments." << std::endl;

  // create runnables
  FbackSender fbsender;
  DataReceiver dreceiver(fbsender); // pass in fbsender so it can tell it a

  Runner& runner = Runner::getInstance(); // get runner

  // add runnables to runner
  runner.addRunnable(dreceiver);
  runner.addRunnable(fbsender);

  runner.run(); // actually run them runnables

  LOG_CLOSE();

  return 0;
}
