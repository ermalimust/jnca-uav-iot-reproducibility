/**
 * Implements class that listens for changes in GPS information and prints them to the standard output.
 *
 * Rui Meireles  {@vassar.edu} 2021, 2024
 */

#include <iostream> // std::cout, std::cerr, std::endl
#include <string>   // std::string
#include <ios>      // std::streamsize
#include <iomanip>  // std::setprecision

#include "../GpsInfo.hpp"                                    // struct GpsInfo
#include "../../util/thread/Runner.hpp"                      // class Runner
#include "../../util/log/LogFile.hpp"                        // LOG_*
#include "../../wichoice/wichoicemaker/mobility/MobiKey.hpp" // class Mobikey
#include "../../wichoice/wichoicemaker/mobility/MobiKeyFactory.hpp" // class
                                                               // MobiKeyFactory

#include "WcGpsPrinter.hpp" // class WcGpsPrinter

#define LOG_FNAME "/var/log/wcgpsprinter.log"

#define NPRINTS_DEF -1 // -1 means infinite

/**
 * Constructor.
 *
 * @param: the number of times to print the GPS information.
 */
WcGpsPrinter::WcGpsPrinter(unsigned long long nprints) : nprints(nprints),
                                                         gpsInfoReader() { }

/**
 * Executes a loop of reading GPS information and printing it to the standard output, until we've printed
 * the requested number of times or the endProgram flag becomes true (through some external intervention).
 *
 * Overrides Runnable::run().
 * Meant to be run as a thread.
 */
void WcGpsPrinter::run() {

  const std::streamsize ogprec = std::cout.precision(); // original precision

  // print header line with format
  std::cout << "gpstime, systime, lat, lon, alt, speed, head, fix, nsats, hdop, mobikey"
    << std::endl;
  
  // main loop
  MobiKeyFactory mobiKeyFactory;
  GpsInfo gpsInfo;
  unsigned long long niters = 0;
  while (!this->endProgram.load()) {

    this->gpsInfoReader.getGpsInfoOnUpdate(gpsInfo); // will block until update

    // if we were awaken due to the program ending we want to stop immediately
    if (this->endProgram.load() || !this->gpsInfoReader.isLive()) break;
    
    MobiKey mobiKey = mobiKeyFactory.create(gpsInfo);
    
    std::cout << gpsInfo.gpstime << ", " <<
                 gpsInfo.systime << ", " <<
                 std::setprecision(10) << // increase precision for lat and lon
                 gpsInfo.lat << ", " <<
                 gpsInfo.lon << ", " <<
                 std::setprecision(ogprec) << // restore default precision
                 gpsInfo.alt << ", " <<
                 gpsInfo.speed << ", " <<
                 gpsInfo.head << ", " <<
                 (unsigned) gpsInfo.fix << ", " <<
                 (unsigned) gpsInfo.nsats << ", " <<
                 gpsInfo.hdop << ", " <<
                 mobiKey.toString() << std::endl;

    if (++niters == this->nprints) break; // we've printed as much as we needed
  }

  // no cleanup needed
}

/**
 * Halts the thread. The superclass one isn't sufficient because the thread may be blocked waiting for
 * a condition to be updated. We must therefore trigger that update manually.
 */
void WcGpsPrinter::stop() {
  this->Runnable::stop();                // sets endProgram flag to true
  this->gpsInfoReader.notifyListeners(); // wake listener up
}

// WcGpsPrinter class implementation end

/**
 * The procedure that actually bootstraps the program.
 * 
 * @param argc: number of command line arguments, including the program's name.
 * @param argv: array of command line arguments, as strings.
 * @return 0 on success, other value on error.
 */
int main(int argc, char *argv[]) {

  LOG_INIT(LOG_FNAME, "wcgpsprinter");

  LOG_MSG("wcgpsprinter starting");
  
  unsigned long long nprints = 0; // 0 means infinite
  if (argc >= 2){
    std::string arg = argv[1];
    try {
      std::size_t pos;
      nprints = std::stoull(arg, &pos);
      if (pos < arg.size())
        std::cerr << "Ignoring trailing characters after #prints: " << arg <<
          std::endl;
    } catch (std::invalid_argument const &ex) {
      std::cerr << "Invalid #prints: " << arg << ". Running limitless." <<
      std::endl;
    } catch (std::out_of_range const &ex) {
      std::cerr << "#iters out of range: " << arg << ". Running limitless." <<
      std::endl;
    }
    
    if (argc > 2) // we ignore spurious arguments
      std::cerr << "Ignoring all command-line arguments beyond the first."
                << std::endl;
  }

  WcGpsPrinter gpsPrinter(nprints); // create runnable

  Runner& runner = Runner::getInstance(); // get runner
  runner.addRunnable(gpsPrinter);        // add runnable to runner
  runner.run();                          // actually run the runnable

  LOG_CLOSE();

  return 0;
}
