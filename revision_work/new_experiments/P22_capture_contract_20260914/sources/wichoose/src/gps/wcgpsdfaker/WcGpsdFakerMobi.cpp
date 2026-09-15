/**
 * Class that implements a GPS daemon using fake, preset, data.
 *
 * Rui Meireles (@vassar.edu) 2024
 */

#include <cstring>   // std::strerror()
#include <cerrno>    // errno
#include <string>    // std::string, std::getline()
#include <fstream>   // std::ifstream
#include <stdexcept> // std::invalid_argument
#include <cstdint>   // uint*_t
#include <exception> // std::exception
#include <sstream>   // std::stringstream, std::endl
#include <iomanip>   // std::setprecision

#include "../../util/log/LogFile.hpp" // LOG_*

#include "WcGpsdFakerMobi.hpp" // class WcGpsdFakerMobi

/**
 * Constructor. Initializes all the necessary state.
 *
 * @param mobiFname: mobility trace filename.
 */
WcGpsdFakerMobi::WcGpsdFakerMobi(std::string const& mobiFname) :
                                            WcGpsdFakerFix(),
                                            mobiTrace(loadMobiFile(mobiFname)),
                                            mobiTraceItr(mobiTrace) { }

/**
 * Empty destructor.
 */
WcGpsdFakerMobi::~WcGpsdFakerMobi() { }

/**
 * Move along the mobility trace exactly one step.
 */
void WcGpsdFakerMobi::stepMobiTrace() {
  this->curGpsInfo = this->mobiTraceItr.next(); // copy and step
}

/**
 * Helper that loads mobility trace from file.
 *
 * @param mobiFname: mobility trace filename.
 * @return the loaded mobility trace.
 */
WcGpsdFakerMobi::GpsInfoVec WcGpsdFakerMobi::loadMobiFile(
                                                std::string const& mobiFname) {
  
  WcGpsdFakerMobi::GpsInfoVec mobiTrace; // new empty trace
  
  std::ifstream mobiFile(mobiFname, std::ifstream::in);
  
  if (!mobiFile.good()) { // does the file exist and can we read from it?
    std::stringstream ss;
    ss << mobiFname << " - " << std::strerror(errno);
    throw std::invalid_argument(ss.str());
  }
  
  std::string line;  // text holder
  try {
    
    // read file line by line
    while (std::getline(mobiFile, line)) {
      
      // skip empty or commented lines (starting with #)
      if (line.size() == 0 || line.rfind("#", 0) == 0) continue;
      
      // parse read line
      std::istringstream iss(line); // convert to stream
      iss.exceptions(std::istringstream::failbit | std::istringstream::badbit);
      
      // line format: lat lon speed head
      float lat, lon, alt, speed, head;
      iss >> lat >> lon >> alt >> speed >> head;
      
      // add to list
      mobiTrace.emplace_back(0 /*systime*/,
                             0 /*gpstime*/,
                             3 /*fix*/,
                             20 /*nsats*/,
                             0 /*hdop*/,
                             lat,
                             lon,
                             alt,
                             speed,
                             head);
    }

  } catch (std::exception const& e) { // problem reading from file
    
    std::stringstream ss;
    ss << "WcGpsdFakerMobi::loadMobiFile() mobility trace reading error: "
       << e.what() <<
       ", line: " << line << std::endl;
    LOG_FATAL_EXIT(ss.str().c_str());
  }
  
  // check mobi trace size
  if (this->mobiTrace.size() == 0) {
    std::stringstream ss;
    ss << mobiFname << " doesn't contain any mobility datapoints";
    throw std::invalid_argument(ss.str());

  }
  
  mobiFile.close(); // file no longer needed
  
  return mobiTrace;
}

/**
 * Constructor method.
 *
 * @param gpsdFakerMobi: the gps daemon to the notified everytime a mobility trace step needs to be
 *                       taken.
 */
WcGpsdFakerMobi::MobiStepper::MobiStepper(WcGpsdFakerMobi& gpsdFakerMobi)
                                              : gpsdFakerMobi(gpsdFakerMobi) { }

/**
 * Executes a loop that takes a step on the mobility trace everytime the user hits enter.
 *
 * Overrides Runnable::run().
 * Meant to be run as a thread.
 */
void WcGpsdFakerMobi::MobiStepper::run() {

  const std::streamsize ogprec = std::cout.precision(); // original precision
  
  const GpsInfo& curGpsInfo = this->gpsdFakerMobi.getCurGpsInfo();

  std::string input;
  do {
    this->gpsdFakerMobi.stepMobiTrace(); // take a step on the trace
    
    // print current position
    std::cout << "Current mobility: " <<
    std::setprecision(10) << // increase precision for lat and lon
    "lat=" << curGpsInfo.lat <<
    ", lon=" << curGpsInfo.lon <<
    std::setprecision(ogprec) << // restore default precision
    ", alt=" << curGpsInfo.alt <<
    ", speed=" << curGpsInfo.speed <<
    ", head=" << curGpsInfo.head <<
    std::endl;;
    
    // wait for user input before taking a step
    std::cout << "Press Enter to move to next datapoint...";
    std::getline(std::cin, input);

  } while ((!this->endProgram.load())); // until SIGKILL do us part
  
}
