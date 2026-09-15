/**
 * Implements class that prints the interface selection to the standard output, everytime a new one is made.
 *
 * Rui Meireles  {@vassar.edu} 2021, 2024
 */

#include <cstdint>  // uint*_t
#include <iostream> // std::cout, std::endl
#include <string>   // std::string
#include <sstream>  // std::stringstream
#include <iostream> // std::cerr, std::endl

#include "../../util/thread/Runner.hpp" // class Runner
#include "../../util/log/LogFile.hpp"   // LOG_*
#include "../shm/WiChoiceReaderMes.hpp" // class WiChoiceReaderMes
#include "../shm/WiChoiceReaderEst.hpp" // class WiChoiceReaderEst

#include "WiChoicePrinter.hpp" // class WiChoicePrinter

#define LOG_FNAME "/var/log/wichoiceprinter.log"


/**
 * Empty constructor.
 *
 * @param wiChoiceReader: where to get wichoice information from.
 * @param printTag: the string to tag each printed wichoice with.
 * @param printHeader: boolean indicating whether we should print a header at the start or not.
 */
WiChoicePrinter::WiChoicePrinter(WiChoiceReader& wiChoiceReader,
                                 std::string const& printTag,
                                 const bool printHeader) :
                                 wiChoiceReader(wiChoiceReader),
                                 printTag(printTag) {
                                   
  if (printHeader) // print header line with format
    std::cout << "tag, gpstime, iface" << std::endl;
}

/**
 * Executes a loop of reading new interface information until the endProgram flag becomes true
 * (through some external intervention).
 *
 * Overrides Runnable::run().
 * Meant to be run as a thread.
 */
void WiChoicePrinter::run() {

  bool firstIter = true; // 1st iteration?
  uint32_t tstampPrev = 0;
  while (!this->endProgram.load()) { // main loop
    
    // get iface choice
    // on 1st iteration get choice immediately to institute a baseline
    // afterwards wait for a choice change before printing
    const WiChoice wiChoice = firstIter ? (firstIter = false,
                              this->wiChoiceReader.getWiChoiceNow()) :
                              this->wiChoiceReader.getWiChoiceOnUpdate();
    
    // stop immediately if either:
    // 1. awoken due to the program ending
    // 2. awoken due to shm destruction (shm no longer live)
    if (this->endProgram.load() || !this->wiChoiceReader.isLive()) break;
    // the notification could be to wake someone else up (e.g., wcsender)
    // in that case, don't reprint
    else if (wiChoice.tstamp == tstampPrev) continue;
    
    // print it out
    std::stringstream ss; // first onto a string stream to ensure separation
    ss << printTag << ", "
       << wiChoice.tstamp << ", "
       << wiChoice.iface;
    std::cout << ss.str() << std::endl;

    tstampPrev = wiChoice.tstamp; // update prev tstamp
  }
  // no cleanup needed
}

 /**
  * Halts the runnable. The superclass one is insufficient because the thread may be blocked waiting for
  * the wichoice to be updated. We must therefore trigger that update manually.
  */
void WiChoicePrinter::stop() {
  this->Runnable::stop();                 // sets endProgram flag to true
  this->wiChoiceReader.notifyListeners(); // wake listener up
}

// WiChoicePrinter class implementation end

/**
 * The procedure that actually bootstraps the program.
 */
int main(int argc, char *argv[]) {

  LOG_INIT(LOG_FNAME, "wichoiceprinter");

  LOG_MSG("wichoiceprinter starting");
  
  if (argc > 1)
    std::cerr << "Ignoring all command-line arguments." << std::endl;

  // create measurement-based wichoice printer
  WiChoiceReaderMes wiChoiceReaderMes;
  WiChoicePrinter wiChoicePrinterMes(wiChoiceReaderMes /* wiChoiceReader */,
                                     "mes" /* printTag */,
                                     true /* printHeader */);

  // create estimate-based wichoice printer
  WiChoiceReaderEst wiChoiceReaderEst;
  WiChoicePrinter wiChoicePrinterEst(wiChoiceReaderEst /* wiChoiceReader */,
                                     "est" /* printTag */,
                                     false /* printHeader */);

  Runner& runner = Runner::getInstance();  // get runner
  runner.addRunnable(wiChoicePrinterMes); // add runnable (measurements)
  runner.addRunnable(wiChoicePrinterEst); // add runnable (estimates)

  runner.run();                          // actually run them runnables

  LOG_CLOSE();

  return 0;
}
