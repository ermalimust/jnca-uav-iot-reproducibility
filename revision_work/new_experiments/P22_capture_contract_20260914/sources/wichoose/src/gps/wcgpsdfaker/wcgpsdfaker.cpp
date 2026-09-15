/**
 * Runs GPS daemon fed by fabricated information.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <memory>    // std::unique_ptr, std::ptr
#include <stdexcept> // std::invalid_argument
#include <iostream>  // std::cerr, std::endl

#include "../../util/log/LogFile.hpp"   // LOG_*
#include "../../util/thread/Runner.hpp" // class Runner
#include "../GpsDaemon.hpp"             // class GpsDaemon

#include "WcGpsdFakerFix.hpp"  // class WcGpsdFakerFix
#include "WcGpsdFakerMobi.hpp" // class WcGpsdFakerMobi

#define LOG_FNAME "/var/log/wcgpsdfaker.log"

/**
 * The procedure that actually bootstraps the program. If an argument is provided, it is treated as the
 * name of a file containing a mobility trace to replay. Otherwise the fixed GPS parameters in the configuration
 * file are used.
 *
 * @param argc: number of command line arguments, including the program's name.
 * @param argv: array of command line arguments, as strings.
 * @return 0 on success, other value on error.
 */
int main(int argc, char *argv[]) {

  LOG_INIT(LOG_FNAME, "wcgpsdfaker");

  LOG_MSG("wcgpsdfaker starting");

  Runner& runner = Runner::getInstance(); // get runner
  
  // must use pointer as no abstract-type variables allowed in c++
  std::unique_ptr<GpsDaemon> gpsd;

  if (argc >= 2) { // we have a mobility trace to work with

    const std::string mobiFname = argv[1]; // implicit std::string conversion

    try {
    
      gpsd = std::make_unique<WcGpsdFakerMobi>(mobiFname);
     
      // add mobility stepper
      WcGpsdFakerMobi& gpsdFakerMobiRef = dynamic_cast<WcGpsdFakerMobi&>(*gpsd);
      WcGpsdFakerMobi::MobiStepper mobiStepper(gpsdFakerMobiRef);
      runner.addRunnable(mobiStepper);
      
    } catch (std::invalid_argument& e) {

      std::cerr << "Invalid argument: " << e.what() << std::endl;
      LOG_CLOSE();
      return 1;
    }
    
    if (argc > 2) // we ignore spurious arguments
      std::cerr << "Ignoring all command-line arguments beyond the first."
                << std::endl;
    
  } else gpsd = std::make_unique<WcGpsdFakerFix>(); // fall back to fixed pos

  runner.addRunnable(*gpsd);              // add runnable to runner
  runner.run();                           // actually run the runnable
  
  LOG_CLOSE();

  return 0;
}
